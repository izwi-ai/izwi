# Izwi distributed serving: implementation plan for the cache-coherent fleet

**Purpose:** An implementation handoff for a coding agent working in the Izwi repository. This plan is the successor to `IZWI_PRODUCTION_SERVING_IMPLEMENTATION_PLAN.md` (the "serving plan"). That plan's software scope is implemented on branch `production-serving`; this document defines the next ten features that turn the worker fleet into a production LLM serving system.
**Prepared:** 23 September 2026.
**Baseline:** Branch `production-serving` @ `9fe4506d` (106 commits over merge-base `7804eee5`). All file anchors below were verified against that revision — re-verify before editing; line numbers will drift.
**Status:** Proposed phases and acceptance criteria. No feature below is claimed to exist yet.
**Hardware requirement:** every feature in this plan must work on all three supported backends — CPU, Apple Metal (unified memory), and NVIDIA CUDA. Backend-specific performance features (CUDA graphs, FP8) are capability-reported, never assumed (INV-08); execution evidence is per-lane. The development host is Apple Silicon, so CPU and Metal lanes are locally executable and evidence-able; CUDA is code-complete + unit-tested locally, with execution evidence hardware-gated and recorded `not run` until a CUDA host is available.

---

## 1. What already exists (reuse map — do not rebuild)

A coding agent must treat the following as **built and tested** on the baseline branch. Each new phase extends these; none of them may be removed or bypassed.

| Asset | Location (verified anchors) | Notes |
|---|---|---|
| Gateway (hardware-free control plane) | `crates/izwi-server/src/gateway.rs:795-827` mounts only `POST /v1/chat/completions` + health/admin/metrics; unmigrated routes 404 fail-closed | Gateway mode constructs no engine (`src/lib.rs:533-604`) |
| Versioned worker protocol | `crates/izwi-serving-protocol` (`types.rs`, `ndjson.rs`, `identity.rs`) | Major-version fenced; bounded NDJSON parsing (256 KiB lines / 16 MiB total / 4096 events) |
| Bounded transport client | `crates/izwi-serving-client/src/lib.rs` (phase deadlines `:211-232`, loopback-plaintext restriction `:325-343`, TLS `:26-172`) | Deterministic `MockWorker` in `src/mock.rs`; 23 contract tests in `tests/http_contract.rs` |
| Real worker | `crates/izwi-serving-worker/src/lib.rs` (atomic admission gate `:1004-1170`, teardown-fenced permit `:1413-1427`), `runtime.rs` wraps `RuntimeService` | Real-process test: `tests/real_cpu_process.rs` |
| Worker registry + routing | `crates/izwi-server/src/worker_registry.rs` (`select_and_reserve_with_fleet_at` `:614-728`, `eligible_deployment` `:1035-1101`) | Capacity-weighted score, randomized tie-breaking, freshness TTL, incarnation/generation fencing, circuit breaker |
| Chat dispatch + retry rules | `crates/izwi-server/src/app/remote_chat_dispatch.rs` (retry allowlist `:476-492`, one alternate `:307-364`) | Retry only on provably-unaccepted; no replay after partial output |
| Deployment placement | `crates/izwi-server/src/gateway_deployments.rs` (`approve_replica` `:160-217`, task-keyed pools `:219-227`) | Replica contract equality enforced |
| Fleet coordination store | `crates/izwi-server/src/batch_runtime/fleet.rs` (claims `:217-255`, self-release `:284-304`), `gateway_fleet.rs` (1/N partition), `app/fleet_coordinator.rs` | SQLite-validated; PG/MySQL SQL present but never executed |
| Node supervisor | `crates/izwi-serving-supervisor/src/{config,launch,lifecycle,locks,main}.rs` | Validate-only `main.rs:733`; canary `main.rs:766`; node-wide model-load lease `locks.rs:63-69`; **CPU lanes only** (`main.rs:638-651`) |
| Paged KV + prefix-cache infra | `crates/izwi-core/src/kv/` + `kv/v2/` (block tables `batch.rs`, arenas `resolved.rs`, residency `residency.rs`); `enable_prefix_caching`, `managed_prefix_cache_salt`, `max_prefix_cache_pages` wired in `runtime/service.rs:2986-2988`; counters `:7221-7231` | Committed cross-request reuse **disabled** for hybrid-attention chat models (see DS1) |
| Engine scheduler | `crates/izwi-core/src/engine/scheduler/` (policies FCFS/Priority/WeightedFair; `max_batch_size`, `max_tokens_per_step`) | Continuous batching (`NativeBatchMode::Continuous`), incremental prefill (`engine/execution.rs:422+`) |
| CUDA graphs / FP8 / MTP | `kernels/cuda/graphs.rs`, `graphs_region.rs` (TensorIsland capture/replay), `kernels/cuda/fp8.rs`, `sampling.rs`; MTP knobs `performance.rs:67-70`, qwen38 cache contract `models/architectures/qwen38/cache.rs` | MTP is single-stream oriented today |
| Chat API surface | `crates/izwi-server/src/api/openai/chat/completions.rs:56-96` (tools, `n`, stream_options, penalties); usage `:853-854` (prompt/completion only) | No logprobs, no cached_tokens, no grammar-constrained `response_format` |
| Boundary gate | `scripts/ci/check-serving-boundary.sh` | Protocol/client/supervisor must stay accelerator-free; worker owns `izwi-core` |
| Bench harness | `scripts/bench/run-gateway-chat-benchmark.py` (+ unit tests) | Closed-loop TTFT/latency/rejections; no live runs yet |
| Operator docs | `docs/dev/PRODUCTION_SERVING_{DISCOVERY,SUPPORT_MATRIX,SINGLE_NODE_RUNBOOK,RELEASE_CHECKLIST,PACKAGING}.md`, route migration ledger in the discovery doc | Support matrix keeps hardware lanes "not run" |

**Known carry-over gaps from the branch review** (closed in DS0): no process-level gateway-only test; single shared API key; no consolidated delivery report; no status-poll jitter; no `max_parallel_model_loads` knob; supervisor refuses Metal/CUDA lanes; support matrix multi-gateway row stale.

## 2. How to work in this repository (agent instructions)

1. Read `AGENTS.md` first. Enter plan mode for any non-trivial task; write a session entry (plan + review sections) to `tasks/todo.md`; record corrections in `tasks/lessons.md`.
2. One phase per work session. Small, reviewable commits in the repo's conventional style (`feat(serving): ...`, `test(serving): ...`, `docs(serving): ...`). Never mix phases in one commit.
3. Before declaring done, run at minimum: `cargo fmt --check` on touched crates, `cargo clippy -p <crate> --lib --tests -- -D warnings` (pre-existing denies in `worker_registry.rs` etc. are documented in `tasks/todo.md:18300-18306` — do not fix unrelated denies silently), the touched crates' test suites, and `scripts/ci/check-serving-boundary.sh` whenever a `Cargo.toml` dependency changes.
4. **Test environment quirks** (verified 2026-09-22): builds need `DEVELOPER_DIR=/Library/Developer/CommandLineTools`; the client contract tests run **only** with `cargo test -p izwi-serving-client --features mock-worker` (bare runs silently execute 0 tests); `izwi-server` test links occasionally fail transiently on `cc` and succeed on retry.
5. Evidence rules (inherited from the serving plan §13/§16/§19): mock-worker success is not real-model evidence; compilation is not execution; unavailable hardware lanes are recorded `not run`, never `passed`; all performance numbers go into a committed benchmark manifest with environment metadata; the support matrix is updated whenever a lane's status changes.
6. Never break: the local/desktop product and legacy local-server profile, the accelerator-free gateway boundary, existing test suites (`izwi-server` lib ≈ 675 tests, serving crates ≈ 100), or the documented public API. Removing tests to get green is forbidden.
7. When an existing implementation differs from this plan, record a short decision in the session entry and preserve the behavioral requirement. Do not restart architecture discussions already settled here.

## 3. Distributed invariants (DINV) — in addition to the serving plan's INV-01…INV-20

| ID | Invariant |
|---|---|
| DINV-01 | Routing uses only authenticated, fresh, bounded status signals. Missing or stale cache/satellite signals degrade to today's capacity-weighted behavior — never to no-admission and never to unbounded trust. |
| DINV-02 | Prefix-cache reuse never crosses tenant scope (salt enforced end-to-end) and never serves state from a different model generation or deployment ID. |
| DINV-03 | Affinity/pinning is best-effort: loss of a pinned owner must fail over through normal worker admission, never surface as a hard error while an eligible worker exists. |
| DINV-04 | Session ownership per realtime stage is exclusive and fenced by incarnation/generation; owner loss produces an explicit interruption, never silent migration. |
| DINV-05 | Host-offloaded KV pages are charged against the same node memory budget as device pages; offload can never exceed the supervisor's allocation. On unified-memory backends (Apple Metal) device and host tiers share one physical pool (INV-10): tiering there manages working-set residency, not additional capacity, and Metal + CPU workers on the same node never each budget the full installed memory. |
| DINV-06 | Multi-gateway quota is either shared-atomic or partitioned — never a mixture; store outage degrades to worker-authoritative admission without panic (extends existing fleet behavior). |
| DINV-07 | A rollout operation never exposes two model generations of one deployment simultaneously, and abort restores the previous generation's eligibility without restart of unaffected workers. |
| DINV-08 | Autoscaling is bounded (min/max per deployment), never flappy (stabilization windows), and scale-down only after drain confirms zero active work. |
| DINV-09 | Every engine lane (native izwi-core, future vLLM lane) satisfies the same worker contract tests; public API behavior is identical across lanes (extends INV-01). |
| DINV-10 | Protocol evolution is additive-only: new status/request fields must be tolerated as absent by both sides; changing semantics of existing fields requires a protocol major version bump. |

---

## 4. Phased backlog

Execution order and dependency graph:

```text
DS0 ─→ DS1 ─→ DS2 ─→ DS7
         │  ╲──→ DS4        DS9a independent of DS1
         │       ╲─→ DS9(cached_tokens)
DS3 (independent)      DS5 (independent)      DS6 (independent)
DS8 (decision gate; spikes may start any time after DS2)
DS10 (register only — no build without entry criteria)
```

---

### Phase DS0 — Foundations and carry-over hardening

**Dependencies:** none. **Outcome:** review debts from the serving branch closed; measurement baseline established.

- [ ] DS0.1 Process-level gateway-only test (serving-plan T01): a test that spawns the `izwi-server` binary in `--role gateway` mode (`CARGO_BIN_EXE_izwi-server`), hits `/livez` and `/readyz`, asserts an inference request is served through an approved mock worker, and — negative control — asserts startup proceeds on a host with no accelerator while no engine is constructed (assert via existing startup logs/metrics, e.g. absence of model-load records).
- [ ] DS0.2 Consolidate the plan-Section-19 delivery report for the serving branch into `docs/dev/PRODUCTION_SERVING_DELIVERY_REPORT.md` using the template (baseline/resulting revision, phases, tests executed/not executed, limitations, rollback, next task). Source material: `tasks/production-serving-status-analysis-2026-09-16.md`, `tasks/todo.md` sessions 2026-09-16…19, support matrix.
- [ ] DS0.3 Status-poll jitter (serving plan §5.2): add bounded deterministic jitter (±10% of `IZWI_GATEWAY_WORKER_STATUS_POLL_MS`) to per-worker poller scheduling in `crates/izwi-server/src/lib.rs:782-804`, seeded per worker for test determinism; boundary tests updated.
- [ ] DS0.4 `max_parallel_model_loads` knob: replace the implicit node-wide exclusive model-load lease (`crates/izwi-serving-supervisor/src/locks.rs:63-69`) with a bounded semaphore honoring `max_parallel_model_loads` from node config (schema v2, default 1 — behavior unchanged by default); add config validation (`≥1`) and a concurrency test proving the bound holds. Decide + record whether the lease must remain exclusive for CPU memory-spike safety; if so, document why in config docs.
- [x] DS0.5 Scoped credentials (serving plan §9.2): introduce per-principal API keys on the gateway — keys carry `principal_id`, role set (`inference`/`admin`/`metrics`), and optional tenant scope; stored as salted hashes in the existing durable store, resolved via bounded `env:`/file refs at boot for bootstrap. Tenant rate quotas and concurrency leases key off the authenticated principal's tenant scope (already derived in `api/request_context.rs:48-62`). Backward compat: the single shared `IZWI_GATEWAY_API_KEY` remains valid as the bootstrap root principal. (Completed 2026-09-25: `gateway_principal_keys.rs` adds bounded scoped keys — principal id, role set `inference`/`admin`/`metrics`, optional tenant scope — provisioned at boot from a fail-closed JSON manifest (`IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST`) whose `key_ref` entries must be bounded `env:`/`file:` references (inline material rejected). Only salted HMAC-SHA256 digests are persisted in a new `gateway_principal_keys` durable-store table; keys are uniform-random bearer tokens, so a fast MAC (not an argon2-class KDF) keeps per-request verification in microseconds. The middleware tries the perimeter root key first — byte-identical backward compatibility — then the directory; the inference route requires the `inference` role (403), and drain/metrics open when the dedicated key is configured OR a scoped principal carries the role (404 when neither, 401 on wrong credentials). Tenant quotas and concurrency needed zero changes: they already key off the authenticated principal's namespace. With the manifest unset the durable store is never opened and behavior is unchanged; all 703 izwi-server lib tests and both gateway-only process legs pass.)
- [ ] DS0.6 Multi-lane supervision (requirement, not a decision): lift `require_cpu_only` (`crates/izwi-serving-supervisor/src/main.rs:638-651`) so the supervisor launches and supervises Metal and CUDA workers exactly as it does CPU workers. The building blocks already exist — config validation for Metal/CUDA assignments (`config.rs:178-254`), per-child device env mapping (`launch.rs:354-403`), and backend identity verification in `izwi-core` (`DeviceSelector::select_assigned`, `device.rs:788-840`) — the work is: remove the gate, wire readiness verification for each lane (Metal device identity, CUDA UUID match reported by the worker descriptor), add a supervisor integration test per lane using fake worker binaries (pattern: `tasks/ci-fake-bin/`), and validate a real Metal worker **on this host**. CUDA launch supervision is validated by unit/integration tests with fake binaries; real CUDA execution evidence stays hardware-gated and `not run`. Record the change in an ADR (`docs/dev/adr/`) and update `PRODUCTION_SERVING_SUPPORT_MATRIX.md`: Metal moves from "not supported as a supervised profile" to "supported on macOS, locally validated"; CUDA supervision "code-complete, execution evidence pending hardware".
- [ ] DS0.7 Baseline measurement (per lane): run `scripts/bench/run-gateway-chat-benchmark.py` against the gateway + one real worker on **CPU** and, after building izwi-core with the Metal feature, **Metal** on this host; commit manifests under `benchmarks/manifests/` with environment metadata (device identity, OS, build flags). These are the before-numbers for DS1/DS2 claims per lane. A CUDA baseline is recorded `not run` (no toolchain/device on this host).
- [ ] DS0.8 Backend fixture parity: establish the chat-fixture tolerance harness comparing outputs across CPU/Metal (and CUDA when available) for the standard chat fixture — the KV precision machinery (`crates/izwi-core/src/backends/kv/precision_tests.rs`) is the pattern. This becomes the standing "backend quality" gate (serving-plan §16.3) each later phase must not regress.

**Acceptance:** all items merged; `cargo test` suites green including the new process test; delivery report committed; benchmark manifest committed.

---

### Phase DS1 — Committed cross-request prefix reuse in the engine

**Dependencies:** DS0. **Outcome:** the engine's biggest throughput lever is safely enabled for supported chat models. (DINV-02.)

**Context:** infra exists (`kv/v2` arenas/block tables; `PrefixPolicy` gating; counters). Blocker on record (`tasks/multi-user-serving-optimizations-plan-2026-08-21.md`, "deliberately NOT done"): for hybrid linear-attention/conv models (qwen38), rebuilding shared spans' recurrent+conv state at a shared prefix boundary is not proven sound.

- [ ] DS1.1 **Conv/recurrent checkpoint-boundary spike** (research, time-boxed): for qwen38 (and LFM2 if applicable), determine whether a shared prefix span's linear-attention and conv state can be rebuilt transactionally when a second request attaches mid-page. Produce: a written analysis in `tasks/`, a prototype test, and a go/no-go. If no-go for a model family, that family keeps per-request KV and gets prefix reuse only at boundaries proven safe (e.g. dense attention models: gemma3, qwen3 dense variants).
- [ ] DS1.2 Dense-model enablement first: flip `PrefixPolicy::CommittedPages` for the dense attention chat family; add concurrency correctness tests — two requests sharing a system prompt, cache hit verified via counters, identical outputs vs cold run (fixture-tolerance comparison), eviction-during-use safety, salt isolation (two tenants, same prompt → no cross hits).
- [ ] DS1.3 Hybrid-model enablement per DS1.1 outcome: implement transactional span rebuild (checkpoint state at page boundaries, rebuild-on-attach under the existing admission gate) or record the family as excluded with rationale in the support matrix.
- [ ] DS1.4 Eviction + accounting: LRU eviction of unreferenced committed pages under arena pressure (reusing existing page-free accounting); expose per-deployment `prefix_hits`, `prefix_queries`, `prefix_evictions`, `reused_tokens` in the engine snapshot (`runtime/service.rs:7221-7231` already has counters — surface them per deployment, not just per process).
- [x] DS1.5 Evidence: rerun the DS0.7 benchmarks with prefix-heavy and prefix-cold workloads **on CPU and Metal** (CUDA `not run` until hardware); commit before/after manifests per lane. Record TTFT delta on shared-prefix workloads. (Executed 2026-09-25 on the tiny hybrid qwen38 fixture through the real gateway: shared lane 40/40 completed with 36 attaches and 4736 avoided-prefill tokens, cold lane 40/40 with 0 attaches and 40 publishes, on both CPU and Metal; manifests `benchmarks/manifests/ds15-{cpu,metal}-{shared,cold}.json` + per-lane summaries. TTFT delta \~0 at fixture scale — 130-token prefill is microseconds, so reuse is counter-proven, not wall-clock-proven; the KV page must stay larger than the ~43-token Xhigh reasoning-instruction block every request carries — see the DS1 analysis doc.)
- [x] DS1.6 Backend parity for reuse: run the DS1.2 correctness suite and the DS0.8 fixture harness on every buildable lane; a prefix-reuse flip that passes on CPU but not Metal is not enabled on Metal (per-backend enablement is recorded in the capability/catalog layer, e.g. `crates/izwi-core/src/catalog/cuda_support.rs` pattern — one model cell per backend, never a blanket flip). (Completed 2026-09-25: `catalog/prefix_reuse.rs` records one cell per family per backend — qwen3.8 hybrid process-parity on CPU+Metal, dense qwen3/gemma3 fixture-suite on CPU only, CUDA not-run, qwen3.5/LFM2/non-chat excluded. The default-on decision: serving surfaces resolve catalog-auto with the evidence-gated table, an explicit `IZWI_ENABLE_PREFIX_CACHING` keeps today's salt-required semantics, an explicit zero is the kill switch, and auto degrades to Disabled with a recorded reason when the page budget cannot fit. Admission evidence: a real worker with no prefix env attaches published prefixes through normal admission (`backend_parity` catalog-auto leg) on CPU, and the Metal lane of the DS1.5 prefix leg stays green on the metal-feature build.)

**Acceptance:** shared-prefix requests demonstrably hit cache (counter evidence + output equivalence); no cross-tenant or cross-generation hits (tests); no regression in the DS0.7 cold-workload baseline beyond noise; support matrix updated per model family.

---

### Phase DS2 — Cache-aware routing and richer status signals

**Dependencies:** DS1. **Outcome:** the registry routes by cache locality, not just load. (DINV-01, DINV-03, DINV-10.)

- [x] DS2.1 **Protocol extension (additive)**: extend `WorkerStatus`/`LoadedDeployment` in `crates/izwi-serving-protocol/src/types.rs` with optional per-deployment fields: `kv_cache_usage_pct`, `prefix_hits_total`, `prefix_queries_total`, `prefix_evictions_total`, `tokens_out_per_s_ema`, `observation_cost_units`. All `Option<T>` with defaults; worker sets them only when the engine reports them; client/registry treat absence as "signal unavailable". Add protocol tests for absent-tolerant decode both directions (DINV-10). No major version bump. (Completed 2026-09-25: protocol minor bumped 0→1; the three registry version gates relaxed to major-only via `SchemaVersion::shares_major_with` so minor-0 workers keep registering — every other boundary already checked major only. Absent/present decode tests plus a registry test proving a (1,0) worker registers, observes, and stays selectable. Mock worker gained a `MockRoutingSignals` knob.)
- [x] DS2.2 Worker exposure: populate the new fields from the engine snapshot in `crates/izwi-serving-worker` status assembly (bounded, fixed-cardinality; no per-request labels). Mock worker: configurable values for tests. (Completed 2026-09-25: KV usage follows the engine's own Prometheus ratio (allocated/capacity pages), queries = hits+misses, evictions direct; worker computes a tokens/s EMA over completed invocations (alpha 0.3) and prices one admission credit per observation, matching its permit model. Wiring proven end-to-end by the real-CPU-process test on a genuine engine.)
- [x] DS2.3 Registry locality scoring: extend `select_and_reserve_with_fleet_at` (`crates/izwi-server/src/worker_registry.rs:614-728`) with a two-stage policy — (a) if any eligible worker's same-deployment `prefix_hits/queries` ratio and kv headroom exceed configured thresholds, prefer it (cache affinity); (b) otherwise fall back to exact current capacity-weighted scoring. Config: `IZWI_GATEWAY_ROUTER_CACHE_AFFINITY=on|off` (default `off` until DS2.5 evidence), thresholds via env with bounded ranges. Unit tests: affinity picks the warm worker; stale/absent signals degrade cleanly (DINV-01). (Completed 2026-09-25: warm candidates compete among themselves on the unchanged cross-multiplied score and win only if one is warm; absent signals, zero queries, or exhausted KV headroom degrade byte-for-byte to the pre-DS2 fallback. Fail-closed on|off parsing plus bounded ratio/KV-usage thresholds.)
- [x] DS2.4 Conversation affinity (best-effort pinning): hash a stable conversation key (first N tokens of the normalized system+history prefix — no raw content leaves the gateway; hash only) into routing; pin to the last worker that served the conversation via a bounded LRU map (`max_entries`, TTL); pinned-but-uneligible → normal admission (DINV-03). Include the key derivation in tests with fixture prompts. (Completed 2026-09-25: the key hashes the whitespace/case-normalized tokens of the system prompt and first user message capped at 256 — a fixed region that never moves as the append-only history grows, which a whole-history prefix hash would not guarantee. Bounded LRU+TTL table in the dispatcher; pins recorded on accepted dispatches including through the alternate failover path; new preferred-selection registry path keeps normal admission on unknown/incarnated or ineligible pins. `IZWI_GATEWAY_SESSION_PIN=on|off` with bounded `_MAX_ENTRIES`/`_TTL_SECS`, default off.)
- [x] DS2.5 Evidence: DS0.7 benchmark extended with a multi-turn workload across 2 workers; compare routing on/off. Commit manifest; flip default only with measured TTFT win. (Completed 2026-09-25: `run-gateway-chat-benchmark.py` gained the multi_turn workload with real assistant replies replayed into the history and per-turn capacity-rejection retries; `run-ds2-routing-benchmark.sh` runs 8 conversations × 4 turns across two workers per lane, one gateway per leg. CPU: attaches 15→19, hits 35→46, avoided-prefill tokens 3520→4288, TTFT p50 60.6→42.2ms. Metal: attaches 11→20, hits 24→48, TTFT p50 327→310ms. Pinning balances workers evenly. Manifests `benchmarks/manifests/ds2-{cpu,metal}-*`. **Default stays off**: the counter evidence is robust but the TTFT direction flipped once across runs at fixture scale (absolute prefill is microseconds), so a production-scale workload must measure a stable TTFT win before the flip.)

**Acceptance:** contract/protocol tests prove additive evolution; routing tests prove locality preference, degradation, and pinning bounds; benchmark manifest shows the affinity win (or the feature stays default-off with documented rationale). (Met: protocol + registry + dispatcher tests all green — izwi-serving-protocol 18, worker_registry 27, remote_chat_dispatch 15, izwi-server lib 688; both lanes' manifests committed with the default-off rationale recorded above.)

---

### Phase DS3 — Realtime voice over the worker boundary

**Dependencies:** none (can start after DS0). **Outcome:** the flagship izwi workload is distributable: realtime ASR/chat/TTS stages execute in workers. (DINV-04.)

**Context:** engine-side realtime execution exists (`engine/execution.rs` `ExecutionMode::Realtime`, stage selectors, realtime preparation modes); local realtime voice lives in `crates/izwi-server/src/app/voice_realtime.rs` (process-local sessions, bounded per §P5.2 work). The worker protocol has no realtime surface (serving plan §6.2 proposed `GET /internal/v1/realtime` WS upgrade — unimplemented).

- [x] DS3.1 Protocol: versioned WebSocket subprotocol (`izwi-realtime-v1`) on the worker: bounded text control frames (JSON: admit/bind/cancel/usage events reusing existing typed types) + binary audio frames (declared format/codec in the admit frame; frame-size caps; order preserved). Bounded: max in-flight frames, byte budget, session cap per worker; reuse NDJSON control types where possible. Add `TaskKind`/capability flags for realtime stages (ASR-stream, TTS-stream) to the descriptor. — protocol minor 2 (`crates/izwi-serving-protocol/src/realtime.rs`): admit/admitted/cancel/ping frames, `RealtimeAudioSpec` (PCM i16 LE mono), fixed 16-byte binary frame codec with hard caps, `RealtimeSessionCloseCode` (4xxx), `WorkerFeature::RealtimeSocket`, `InputFormat::PcmAudio`; protocol absent/present decode tests.
- [x] DS3.2 Worker runtime: session admission reuses the atomic admission gate; one execution owner per stage; audio pushed at safe interruption points; cancellation semantics identical to HTTP path (`CancellationRequested → ExecutionStopping → terminal`); session teardown releases permits only on confirmed teardown (INV-12 semantics). — `/internal/v1/realtime` on speech_to_text workers (`IZWI_WORKER_TASK`); attempts live in the shared table so HTTP query/cancel behave identically; admission reuses the HTTP admit path's fencing (incarnation/deployment/generation/task/readiness/capacity/drain); capacity releases only after the stage's runtime stream drops; task-aware warm-up (chat warm-up, bounded synthetic-utterance ASR warm-up) fails boot closed. TTS stage rejected until implemented.
- [x] DS3.3 Session ownership records: gateway-side bounded session registry mapping session → (worker, incarnation, deployment generation, stage leases); lease expiry with explicit interruption event to the client; reconnect = new session (no promised resume — document). — `RealtimeSessionRegistry` (bounded LRU + TTL) in `crates/izwi-server/src/app/realtime_relay.rs`; owner loss synthesizes an explicit internal-error event before closing; reconnect is always a new session (documented in the relay module doc).
- [x] DS3.4 Gateway relay: realtime voice route in gateway mode upgrades client WS ↔ worker WS per stage; multi-stage voice workflow binds ASR/chat/TTS to workers per deployment pools (reuse `gateway_deployments.rs` task pools); barge-in passes through as control frames where the local product already supports it; never hold two stage permits such that deadlock is possible (plan §10.4 — account stage reservations separately). — `/v1/realtime/ws` behind `IZWI_GATEWAY_REALTIME=on|off` (default off = byte-identical; pinned mode rejects the flag); one tenant concurrency lease + one local dispatch slot per stage session, never held across another stage's admission; relay speaks izwi-realtime-v1 end-to-end (public transcription-realtime envelope translation initially deferred, then delivered as a DS3.4 follow-up — see DS3.9).
- [x] DS3.5 Mock + contract tests: mock realtime worker (delayed/cadenced audio, disconnect mid-stream, cancel races, stale incarnation); contract tests over real WS: ordering, bounds, one-terminal-outcome, owner-loss interruption. Map to serving-plan T21 semantics. — `MockRealtimeKnobs` on the mock worker; 8 T32/T33 contract tests (`izwi-serving-client/tests/realtime_contract.rs`) + 12 worker session tests (`izwi-serving-worker/tests/realtime_ws.rs`) over real sockets: ordering, bounds, one-terminal, owner loss, stale incarnation, duplicate attempt, capacity, draining, HTTP-cancel parity.
- [x] DS3.6 Route migration ledger: update the ledger rows for realtime transcription/voice (currently local-only) — mark "gateway: preview behind flag" only after contract tests pass; keep default local profile until DS3 evidence is complete. Update support matrix honestly (mock evidence ≠ real hardware voice evidence). — ledger row updated after the contract tests passed; support matrix records per-lane realtime evidence honestly.
- [x] DS3.7 Real-evidence slices: one real CPU worker **and one real Metal worker** (this host) executing streaming ASR through the subprotocol with small fixtures; record process-level evidence per lane. Metal is the primary local voice profile (the desktop product's home platform) and must not be second-class in this phase. — ignored test `real_cpu_realtime.rs` (backend via `IZWI_RT_EVIDENCE_BACKEND`); CPU lane: real worker binary + real `Nemotron-3.5-ASR-Streaming-0.6B` artifact, 36 frames of `data/fox.wav` (24 kHz mono) → transcript `"The quick brown fox jumps."`, one Completed terminal, clean close, attempt Completed over HTTP. Metal lane: same test with `IZWI_BACKEND=metal` + device identity (see support matrix for the recorded outcome). Releasing the evidence exposed and fixed a real regression (see `fix(runtime)` commit: Nemotron routed to Engine admission that cannot admit its tensor state; single-node realtime ASR had silently degraded to chunked fallback since 7215003a).
- [x] DS3.8 TTS-stream stage execution (follow-up, 2026-09-26): the protocol was stage-symmetric from minor 2, so no protocol change was needed. Workers now execute the TTS stage end to end — `IZWI_WORKER_TASK=text_to_speech` with a fail-closed TTS-family gate, task-aware bounded warm-up that requires actual PCM (a terminal marker alone proves nothing: the engine streaming path always emits one), `RealtimeTtsStageStream` wrapping `RuntimeService::generate_streaming`, and a session loop that accumulates `Input` text frames against the session byte budget, commits synthesis on `Finish` (empty-text finish mirrors the ASR empty-transcript finish), streams bounded IRTA frames back (oversized chunks split; final flag on the terminal frame; a full outbound queue is treated as a lost peer), and cancels through the identical ladder with teardown confirmed when the synthesis forwarder exits. The client transport serves both stages from one session type (`send_text`/`next_audio`, announced `output_audio` spec, bounded demux queues); the mock worker gained a deterministic TTS script; the gateway relay resolves the admit's task to the optional text_to_speech pool and forwards text/audio verbatim (a missing stage refuses its admits with an explicit PolicyDenied close; boot fails closed only when no realtime stage can be served). Evidence: real CPU + Metal Kokoro-82M workers (133,200 bytes / 2.77 s at 24 kHz per lane; the run also exposed and fixed reader-abort teardown that could turn an orderly close into a TCP reset, costing the client its terminal event).
- [x] DS3.9 Public transcription-realtime envelope translation (DS3.4 deferred item, 2026-09-26): a client of the single-node `/v1/speech-to-text/realtime/ws` surface can repoint at the gateway unchanged. At the realtime route the gateway now dispatches on the subprotocol offer — offered `izwi-realtime-v1` keeps the byte-identical passthrough relay; any other offer (or none) gets the translator (`app/realtime_translate.rs`), which negotiates both public wire modes (legacy `transcription_realtime_v2` JSON and typed `transcription_realtime` v3 envelopes, resume rejected as on the single-node surface), re-encodes public ITRW audio frames onto the worker's IRTA framing with the client's sequence numbers, defers admission to the first audio frame (the public envelope declares the sample rate per frame; a no-audio session finishes locally without dialing), accumulates worker deltas into the replaceable partial hypothesis with replace-on-terminal finals (worker contract: the last pre-terminal delta carries the full final text), answers pings locally, reports v3 audio gaps, and maps owner loss and worker errors onto the public error vocabulary. Session accounting is identical to the passthrough mode (bounded registry, tenant lease, attested caller, session budget); gateway-minted session/request/attempt ids replace the public surface's absent ids. The single-node alias path is mounted next to `/v1/realtime/ws`. Evidence: 9 unit tests (accumulator hold-back/replace semantics, ITRW→IRTA re-encode, sample-rate lock, gap detection, v3 golden shape + finality order, v2 JSON shape, negotiation, error mapping) and two process tests through the real gateway binary (public v2 client at the alias path; public v3 client with envelope continuity, gap detection, and the final/closing/closed ladder). Public realtime-voice envelope translation stays deferred with the voice surface.

**Acceptance:** a voice session traverses gateway → worker WS → engine and back with order preserved and bounds enforced; owner-loss produces explicit interruption; all DINV-04 tests pass; ledger + matrix updated without overclaiming.

---

### Phase DS4 — Hierarchical KV offload (GPU→host)

**Dependencies:** DS1. **Outcome:** cold KV pages spill to host memory instead of forcing preemption/waiting; multi-turn prefixes survive longer. (DINV-05.)

- [ ] DS4.1 Design note: page-granular demotion/promotion on top of `kv/v2` arenas; `KvStorageTier::{Device,Host}` transitions tracked via residency state (`crates/izwi-core/src/kv/residency.rs`); host pool bounded by a new engine budget wired from the supervisor's node memory ledger (same accounting as INV-10/DS0 budgets). **Per-backend semantics are part of the design:** on CUDA (discrete VRAM), host-tier spill adds real capacity across PCIe and is the primary concurrency lever; on Metal (unified memory), there is no additional pool — "offload" means cold-page working-set trimming via residency controls, the budget is the common node pool (DINV-05), and the expected win is retention of multi-turn prefixes plus admission headroom, not raw capacity. Pin the design decision: no cross-process/cross-node KV transfer (serving-plan exclusion stands).
- [ ] DS4.2 Demotion: when device arena pressure crosses a high watermark, evict *unreferenced* committed pages to host (async, bounded in-flight copy budget, cancellation-safe); never evict pages referenced by active requests (fall back to preemption behavior instead).
- [ ] DS4.3 Promotion: on prefix match pointing at host-resident pages, promote before attach within the admission wait budget; miss → cold path. Counters: `kv_host_pages`, `demotions_total`, `promotions_total`, `promotion_latency`.
- [ ] DS4.4 Tests: unit tests for tier transitions and budget enforcement (DINV-05); engine test — concurrent shared-prefix requests with a device arena too small for all, asserting correctness vs cold-run outputs and bounded host usage; no-regression run on the standard baseline.
- [ ] DS4.5 Evidence: benchmark manifest — high-concurrency workload with offload on/off on CPU and Metal lanes (on Metal, success means better retention/admission headroom under the shared budget, per DS4.1 semantics — not more total memory); CUDA lane recorded `not run` until hardware evidence. Counters exposed per DS2.1 field set (additive protocol fields already reserved).

**Acceptance:** offload raises sustainable concurrency in the manifest without correctness regressions; budget never exceeded; counters exposed per DS2.1 field set (additive protocol fields already reserved).

---

### Phase DS5 — Fleet authority: shared admission, validated stores, multi-process rig

**Dependencies:** none. **Outcome:** the multi-gateway profile becomes *validated* rather than best-effort. (DINV-06.)

- [ ] DS5.1 PostgreSQL in CI: add a test profile running the izwi-server store + fleet store against real PostgreSQL (service container or local brew service documented in the runbook); migrate the fleet dialect tests (`batch_runtime/fleet.rs:749-778` string-shape test) into *execution* tests; fix dialect SQL until green; repeat for MySQL if the driver burden is acceptable, else keep honestly marked unvalidated.
- [ ] DS5.2 Multi-process fleet rig: an integration test (or `tests/` harness binary) that launches **two gateway processes + two worker processes** against one shared PG/SQLite store and proves: T07 (atomic admission across gateways), T20 (shared quota not multiplied), P8.1 atomic registry writes (shared approvals/observations), claim steering, gateway-crash + TTL reap recovery, and DINV-06 outage degradation. Prefer one long-running `tests/fleet_rig.rs` using the existing binaries via `CARGO_BIN_EXE_*`.
- [ ] DS5.3 Shared-atomic quota selection: promote the existing `FleetCapacityView` claim path to the default when `IZWI_GATEWAY_FLEET_DB_PATH` is set, with 1/N partitioning remaining as the explicitly-chosen fallback; document both in the fleet ops docs; tests for both paths.
- [ ] DS5.4 T25 process evidence: second-supervisor collision test with real processes (fence blocks until first supervisor's workers exit) — the locks unit tests exist; prove it with two supervisor processes on one node config.
- [ ] DS5.5 Docs: runbook + support matrix updated — multi-gateway moves from "not supported; gates open" to "supported on validated stores" with the store matrix (SQLite standalone / PG fleet) explicit.

**Acceptance:** rig proves all listed scenarios against real processes + real PG; store matrix documented; no silent SQLite-only assumptions remain in fleet-critical code paths.

---

### Phase DS6 — Coordinated blue-green rollout

**Dependencies:** none (builds on existing supervisor + gateway primitives). **Outcome:** one operator action performs a generation cutover with automatic abort. (DINV-07.)

- [ ] DS6.1 Rollout plan type: a declarative rollout spec (new deployment generation + canary worker + promotion policy) accepted by the supervisor and/or gateway CLI; validate before mutation (reuse validate-only machinery).
- [ ] DS6.2 Coordinator state machine: `validate → launch replacement (canary first) → verify readiness (existing ReadinessTracker) → dual-eligible window (both generations in pool, old marked draining-new) → drain old (existing drain_and_stop) → commit → abort path restores previous eligibility at any step`. Persist rollout state in the supervisor for resume-after-crash; two generations of one deployment must never both be *admission-eligible* simultaneously (DINV-07) — enforce in `gateway_deployments.rs` approve path.
- [ ] DS6.3 Gateway-side cutover: replace the manual approvals-file edit step with an atomic approvals update (the shared approvals file machinery from the serving branch already supports versioned views — extend with a guarded write API).
- [ ] DS6.4 Tests: state-machine unit tests (every transition + abort), process test — canary fails readiness → abort leaves old generation serving with zero dropped requests (mock workers); drain completes only after in-flight work terminates (reuses drain tests).
- [ ] DS6.5 Runbook: replace the manual canary section with the new command; document rollback.

**Acceptance:** a scripted end-to-end rollout test passes (promote and abort paths); runbook updated; no manual approvals surgery required.

---

### Phase DS7 — Signal-driven worker autoscaling

**Dependencies:** DS2 (signals), DS6 (drain integration). **Outcome:** the supervisor becomes a capacity manager within explicit bounds. (DINV-08.)

- [ ] DS7.1 Policy config: per-deployment `min_workers`, `max_workers`, `scale_up_queue_depth` (sustained N polls), `scale_down_stabilization_window`; validated ranges; off by default (`min=max=static`) — behavior identical when disabled.
- [ ] DS7.2 Supervisor implementation: scale-up launches additional workers through the existing launch/readiness/canary path; scale-down selects the most-recently-idle worker, marks draining, waits for zero active work (registry observations), then stops via existing `drain_and_stop`. Hysteresis: no scale event within the stabilization window after another scale event.
- [ ] DS7.3 Budget integration: every scale decision respects the node resource ledger (memory/CPU/device exclusivity) — impossible scale-ups are rejected with diagnostics, not launched into OOM.
- [ ] DS7.4 Tests: policy unit tests (signals → decisions, hysteresis, bounds); integration test with mock workers — sustained queue depth scales 1→2, idle window scales 2→1 with drain observed; ledger-overcommit rejected.
- [ ] DS7.5 Docs: runbook section; support matrix notes autoscaling as supervisor-managed (no K8s dependency), fleet profile only.

**Acceptance:** scale events respect bounds, budgets, drain, and hysteresis in tests; disabled-by-default preserves current behavior.

---

### Phase DS8 — Alternative engine lane (vLLM worker), behind a decision gate

**Dependencies:** decision gate after DS2. **Outcome:** model breadth above the per-device ceiling without weakening the worker contract. (DINV-09.)

**Decision gate (record as ADR before any build):** proceed only if a concrete requirement exists for a model that cannot fit izwi's per-device quantized ceiling (e.g. 70B FP8/FP16) or for ecosystem breadth (tokenizers/logprobs/constrained decoding parity), AND DS1/DS2 evidence shows the native lane's remaining gap is breadth rather than performance. **Scope: this lane is a CUDA-profile option** — vLLM has no Metal execution path, so the Metal profile relies on the native engine (quantized GGUF keeps the per-device ceiling high there); if a 70B-class Metal requirement emerges, that is a new decision (e.g. an MLX-based lane), out of scope here. In-engine tensor parallelism in Candle remains rejected (serving-plan exclusion; cost/benefit documented in the analysis).

- [ ] DS8.1 Contract-fit spike (time-boxed): map the worker protocol (`InvocationRequest` → vLLM OpenAI chat completions on loopback; vLLM SSE → `InvocationEvent` stream; usage → usage; cancel → abort). Deliverable: written mapping + a prototype translation layer; identify gaps (logprobs mapping is trivial; constrained output mapping; no incumbent for realtime — the lane is chat-only).
- [ ] DS8.2 `izwi-serving-worker-vllm` binary (new crate, same protocol crate): implements describe/status/invocations/cancel over the same bounded NDJSON contract; status maps vLLM metrics (`num_requests_running/waiting`, `kv_cache_usage_perc`, prefix-cache counters) into the DS2.1 additive fields; admission = worker-local semaphore honoring `max_active_invocations` configured to match vLLM `--max-num-seqs`; attempt/dedup table per contract; TLS/loopback rules identical to the native worker.
- [ ] DS8.3 Boundary + packaging: the new crate must pass `check-serving-boundary.sh` semantics (it may not link accelerator crates — it drives vLLM over HTTP; add it to the gate as an allowed HTTP-only lane with its own negative control); packaging targets + compose example extension.
- [ ] DS8.4 Contract test suite parity: run the existing `http_contract.rs` suite against the vLLM-lane worker (mock vLLM server fixture in tests; no real GPU required for contract conformance) — DINV-09 is literally this test run.
- [ ] DS8.5 Real-evidence slice (hardware-gated): one CUDA host running vLLM with a mid-size model; gateway routes chat to it identically; record evidence; support matrix gains an "engine lane" column.

**Acceptance:** contract parity proven; gateway behavior lane-identical (INV-01 tests extended to assert identical public fixtures across lanes); decision ADR committed regardless of outcome.

---

### Phase DS9 — API completeness for production parity

**Dependencies:** DS9a independent; DS9b benefits from DS1. **Outcome:** OpenAI-compatible serving parity items closed.

- [ ] DS9.1 `usage.cached_tokens` (DS1-dependent): report per-request cached prefix tokens from DS1 counters through `ChatGeneration` usage into public usage + worker events; mock worker configurable for tests. (Metering prerequisite; providers bill cached input at steep discounts — parity matters.)
- [ ] DS9.2 Grammar-constrained `response_format` (research-first, time-boxed): evaluate a compressed-FSM/xgrammar-style constrained sampler over the Candle stack (per-step logit masking at the sampling kernel boundary — `kernels/cuda/sampling.rs` and CPU equivalent). Deliverable: design note + feasibility spike (JSON subset first: valid-JSON mode), then schema→FSM for full JSON Schema. If infeasible at acceptable effort, record the decision and keep `response_format: json_object` validation-only with documented behavior.
- [ ] DS9.3 `logprobs`/`top_logprobs`: surface per-token logprobs through the engine sampling path and stream events (additive protocol fields); public SSE parity with OpenAI semantics.
- [ ] DS9.4 MTP generalization (DS1-dependent): extend Qwen38's single-stream MTP (`performance.rs:67-70`, qwen38 cache contract) to operate within continuous batches where the cache contract permits (the multi-user plan noted solo rows keep MTP — revisit post-DS1); measured acceptance: tokens/s improvement in the benchmark manifest, no ITL regression.

**Acceptance:** each item lands with fixture tests + one public-contract test; benchmark manifest updated where performance-affected.

---

### Phase DS10 — Deferred register (no build without entry criteria)

Maintained as decisions, not tasks:

- **MoE architectures + expert parallelism:** entry criteria — a MoE chat model in the supported catalog AND the DS5 fleet profile validated on real hardware. (Today: no MoE chat families in `models/architectures/` — qwen3/35/38, lfm2, gemma3 are dense.)
- **Prefill/decode disaggregation (PD):** entry criteria — multi-node RDMA-class fabric validated by DS5's rig AND measured ITL-SLO violations that colocated scheduling cannot fix. (Reference implementations: Dynamo, llm-d, Mooncake/NIXL.)
- **Tensor parallelism in-engine:** permanently rejected in favor of DS8's lane approach unless the decision gate flips with new evidence.
- **Cross-node/cross-process KV transfer:** rejected (serving-plan exclusion stands; DS4 stays node-local).

---

## 5. Test and release gates

New test IDs continuing the serving plan's T01–T26 series:

| Test ID | Scenario | Phase | Acceptance |
|---|---|---|---|
| T27 | Two requests share a prefix; second hits committed cache | DS1 | Counter evidence; outputs equivalent; salt isolation holds |
| T28 | Eviction during active reference; concurrent shared-prefix load | DS1 | No corruption; preemption fallback path works |
| T29 | Additive status fields absent/present both directions | DS2 | Protocol decode tolerant; no major bump |
| T30 | Cache-affinity routing with warm + cold workers | DS2 | Warm worker preferred above thresholds; stale signals degrade to capacity scoring |
| T31 | Conversation pinning under owner loss | DS2 | Failover via normal admission; bounded pin table |
| T32 | Realtime session over worker WS: ordering, bounds, one terminal outcome | DS3 | Contract parity with HTTP semantics |
| T33 | Realtime owner loss / stale incarnation | DS3 | Explicit interruption; no silent migration; capacity released on confirmed teardown |
| T34 | KV offload budget + correctness under pressure | DS4 | Host usage bounded by ledger; outputs equivalent |
| T35 | Two gateways + shared store: concurrent admission & quota | DS5 | T07/T20 semantics with real processes |
| T36 | Two supervisors, one node: fence collision | DS5 | Second supervisor blocked until first drains |
| T37 | Blue-green promote + abort | DS6 | Zero dropped requests; no dual-eligible window violation |
| T38 | Autoscale up/down with drain and hysteresis | DS7 | Bounds, budgets, stabilization respected |
| T39 | vLLM-lane contract parity | DS8 | Existing contract suite green against lane worker |
| T40 | `cached_tokens`, logprobs, constrained output public fixtures | DS9 | OpenAI-shape parity |

Hardware matrix additions (record `not run` until executed): CUDA prefix-reuse + offload lane, multi-GPU vLLM lane, multi-node rig on real fabric. **Metal lanes are locally executable on the Apple Silicon development host** and are scheduled as first-class evidence (DS0.6/DS0.7/DS1.5/DS3.7/DS4.5), not deferred; a mixed CPU+Metal worker pool over real HTTP (serving-plan P7.4's local half) is testable on this host and belongs in DS2/DS5 evidence.

## 6. Configuration surface (summary)

- Gateway: `IZWI_GATEWAY_ROUTER_CACHE_AFFINITY`, `IZWI_GATEWAY_ROUTER_CACHE_MIN_HIT_RATIO`, `IZWI_GATEWAY_ROUTER_CACHE_MAX_KV_USAGE_PCT`, `IZWI_GATEWAY_SESSION_PIN`, `IZWI_GATEWAY_SESSION_PIN_MAX_ENTRIES`, `IZWI_GATEWAY_SESSION_PIN_TTL_SECS` (DS2); `IZWI_GATEWAY_PRINCIPAL_KEYS_MANIFEST` (DS0.5); rollout CLI (DS6).
- Supervisor node config (schema v2, TOML): `max_parallel_model_loads` (DS0.4), per-deployment autoscaling block (DS7), host KV pool budget (DS4).
- Worker env: cache-stats emission toggle (DS2.2), realtime subprotocol flags (DS3), KV offload budget (DS4), engine-lane descriptor (DS8).
- All secrets remain `env:`/file references; all new numeric config bounded and validated at startup with actionable diagnostics (serving-plan §14 rules).

## 7. Definition of done and delivery report

Each phase ends with a session report appended to `tasks/todo.md` (commits, decisions, tests run/not run, evidence, next step) and, where applicable, updates to the support matrix, runbook, and benchmark manifests. The programme is complete for a profile only when its gates pass; report profiles individually (development mock/CPU; production single-node; multi-gateway fleet on validated stores; vLLM lane if gated in). Final report reuses the serving-plan §19 template.

**Final acceptance statement (target):** A client uses one stable Izwi API. The gateway routes by cache locality and capacity across a fleet of independently supervised workers; shared prefixes are reused safely within tenant scope; realtime voice sessions execute in workers with explicit ownership; the fleet's quotas stay correct across gateways and store outages; and any engine that satisfies the worker contract can serve traffic without changing the public API.
