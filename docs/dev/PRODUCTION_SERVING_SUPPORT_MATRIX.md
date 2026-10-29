# Production-serving support matrix

Status: exact-profile support statement, not a production-readiness certificate

This matrix applies to Izwi's separated gateway/supervisor/worker architecture.
It does not change the support contract of the existing `--role local` desktop
and CLI workflows. A source path, parser test, scripted-runtime worker, or compilation result
is never promoted to real hardware or performance evidence.

## Current profile cells

| Topology | Gateway | Worker/backend | Model/artifact evidence | Current status |
|---|---|---|---|---|
| One machine, one device | One hardware-independent gateway | One supervised CPU worker | Generated tiny GGUF for the already-supported `LFM2.5-1.2B-Instruct-GGUF` route completed JSON and SSE through a subprocess worker | Development vertical slice proven; not a production model, capacity, soak, or performance certificate |
| One machine, multiple devices | One gateway | Multiple independent scripted-runtime workers | Deterministic chat replicas prove compatible routing, independent request identity, worker-authoritative capacity, retry fencing, and failure isolation | Scripted-runtime evidence only; no physical multi-device claim |
| One machine, Apple silicon | One gateway | Metal worker (metal-feature build, device identity `metal:<registryID>` verified at startup) | **Real Metal inference executed**: 120 gateway-routed generations through a metal worker (debug build, tiny synthetic fixture, `benchmarks/manifests/ds07-metal-baseline.json`); `backend_parity.rs` proves greedy Metal outputs equal CPU outputs on the fixture, including the DS1.5 committed-prefix attach leg (`benchmarks/manifests/ds15-metal-summary.json`); supervisor launch + exact device identity process-evidenced (`multi_lane_launch.rs`) | Development Metal lane proven on the fixture; not a production model, capacity, soak, or performance certificate |
| One machine, NVIDIA | One gateway | CUDA-capable worker binary via `--cuda-worker-binary` | The supervisor accepts, validates, and launches CUDA assignments with exact declared UUID and device-visibility environment (process test `multi_lane_launch.rs`); no CUDA toolchain or NVIDIA device was available in this work session | Supervision code-complete; CUDA execution not run — no CUDA or multi-GPU support claim |
| Multiple machines | One gateway | Versioned node/worker-pinned HTTPS approvals | Fleet topology fails closed without mTLS client identity, HTTPS, and exact operator-pinned node/worker IDs; the worker terminates TLS/mTLS from bounded file references (supervised workers stay loopback-only), but no separate-machine handshake or artifact path was exercised | Not supported as an operational profile |
| Multiple gateways | Two or more gateways | Any worker fleet | Two real gateway processes over one shared coordination database prove T07 (atomic admission), T20 (partitioned quota not multiplied), P8.1 (shared approvals), claim steering, gateway-crash + TTL recovery, and store-outage degradation (`tests/fleet_rig.rs`, SQLite and PostgreSQL lanes); a hung coordination store is unit-tested: status polls and dispatch proceed and admission falls back to the worker. Capacity claims cover only the transit to worker acceptance and steer selection while the worker remains the atomic admission arbiter | Supported on the validated store lanes: SQLite (one host) and PostgreSQL (shared fleet); MySQL is not supported; circuit state and accepted-work ownership stay per-gateway |

The gateway in every cell is the standalone `izwi-gateway` binary (Docker
target `production-gateway`), which links no inference engine; the process
suites run it. `izwi-server --role gateway` runs the same gateway code with
the same flags and stays supported (`gateway_role_process.rs`).

## Worker contract, health, and stream backpressure

These apply to every cell above.

| Behaviour | Status | Evidence |
|---|---|---|
| Worker capability | Derived from the loaded model: precision and execution representation from the served variant, a chat context from the runtime's fitted context, and an output limit up to it. A supervisor passes its declared profile and limits (`IZWI_WORKER_PRECISION`, `_EXECUTION_REPRESENTATION`, `_MAX_CONTEXT_TOKENS`, `_MAX_OUTPUT_TOKENS`); the worker advertises exactly those once the model proves them and fails startup otherwise | Worker unit tests against a supervisor-rendered node config; CPU only |
| Runtime health | A panic after native entry or a latched fatal device fault (CUDA sticky errors, Metal command-buffer errors; out-of-memory stays recoverable) marks the runtime Failed. Worker status then reports `Failed`, the gateway and supervisor stop routing to it, and the worker exits with status 70 (`IZWI_WORKER_EXIT_ON_RUNTIME_FAILURE`, default on) so the supervisor restarts it. A drain-recoverable fault keeps the worker running but not Ready. The local server's `/livez` fails only on Failed, and `/readyz` no longer fails on saturation | CPU fault injection through the runtime's own classifier; the CUDA and Metal fault-text classification is unit-tested only and has not met a real device fault |
| Stream backpressure | The runtime, the worker's NDJSON relay, the local SSE path and the gateway relay merge text deltas for a lagging reader instead of cancelling. A stream fails only when the reader accepts nothing for the slow-consumer timeout (`IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS`, default 30 s; the gateway relay's `IZWI_GATEWAY_SLOW_CONSUMER_TIMEOUT_MS`, default 5 s) or the merged text exceeds its bound. Terminal events never overtake text | Paused-reader tests at each hop, including a gateway reader paused while a worker bursts through both relay hops |

## Committed prefix reuse by model family and backend (DS1.6)

The catalog (`crates/izwi-model-catalog/src/catalog/prefix_reuse.rs`) records one cell
per model family per backend lane; serving surfaces default to catalog-auto,
which engages reuse only for supported cells (normally those with lane
evidence; a cell supported ahead of its evidence says so). An explicit
`IZWI_ENABLE_PREFIX_CACHING=1` bypasses the table (salt required); an explicit
`0` is the kill switch. Auto degrades to Disabled with a recorded reason when
the runtime's page budget cannot fit a safe prefix reserve.

Hybrid chat families (Qwen3.8, Qwen3.6, LFM2.5, LFM3) declare where a prompt's
stable history ends (before the final assistant generation header), snapshot their
recurrent or ShortConv state there, and attach the next turn at that snapshot, so a turn
prefills only the previous answer and the new message. Their snapshots have
their own budget (`IZWI_MAX_PREFIX_SNAPSHOTS`, default 8 rows of recurrent
state), reserved on the model lease at load. A Qwen3.8 load with its MTP head
keeps every state domain private (no reuse).

| Family | CPU | Metal | CUDA |
|---|---|---|---|
| Qwen3.8 chat (hybrid) | Supported — process parity (`backend_parity` prefix leg) | Supported — process parity (metal-feature leg) | Supported ahead of evidence (operator decision) — contract declared; the GPU pre-flight runs `cuda_attached_history_matches_the_fresh_run`; `IZWI_QWEN38_PREFIX_REUSE=0` turns it off |
| Qwen3.6 chat (hybrid MoE) | Supported — fixture suite (attach test bit-identical, three-turn chain) | Not enabled — attach leg passes, opt-in | Supported ahead of evidence (operator decision) — contract declared; the GPU pre-flight runs `cuda_attached_prefill_matches_the_fresh_run` |
| LFM2.5 / LFM3 chat (hybrid ShortConv) | Supported — fixture suite (`lfm2_attached_prefix_continues_exactly_as_a_fresh_run` bit-identical, `lfm_ring_snapshots_publish_and_fork_into_the_attaching_request`); real-weight cut at the 256-token page boundary equals one-shot prefill | Supported — real-weight page-boundary cut equals one-shot prefill on Metal and the snapshot fork test is device-agnostic; the attach test itself runs on CPU | Not enabled — contract declared, no GPU evidence |
| Qwen3 chat (dense) | Supported — DS1.2 fixture suite | Not enabled — contract declared, no lane parity | Not enabled — no lane evidence |
| Gemma3 chat (dense) | Supported — DS1.2 fixture suite | Not enabled — contract declared, no lane parity | Not enabled — no lane evidence |
| Voxtral (LM) | Not enabled — contract declared, no lane parity | Not enabled | Not enabled |
| Nemotron streaming ASR (`izwi-realtime-v1`) | Supported — real process evidence: the real worker binary with the real `Nemotron-3.5-ASR-Streaming-0.6B` artifact streamed `data/fox.wav` (36 frames, 24 kHz) through the WebSocket subprotocol → transcript "The quick brown fox jumps.", one Completed terminal, clean close (`real_cpu_realtime.rs`, ignored test, `IZWI_RT_EVIDENCE_BACKEND=cpu`) | Supported — same test with the metal-feature worker and device identity `metal:<registryID>` verified at startup (`IZWI_RT_EVIDENCE_BACKEND=metal`) | Not enabled — no lane evidence |
| Kokoro realtime TTS (`izwi-realtime-v1` TTS-stream) | Supported — real process evidence: the real worker binary with the real `Kokoro-82M` artifact synthesized the fox sentence over the subprotocol → 133,200 audio bytes (2.77 s at 24 kHz), non-silent, final-flagged terminal frame, one Completed terminal, clean close, HTTP-resolvable attempt (`real_cpu_realtime_tts.rs`, ignored test, `IZWI_RT_EVIDENCE_BACKEND=cpu`; ~41 s wall clock, RTF ≈ 15) | Supported — same test with the metal-feature worker and device identity verified at startup (`IZWI_RT_EVIDENCE_BACKEND=metal`; ~8 s wall clock) | Not enabled — no lane evidence. Other TTS families (Fish S2, Qwen3-TTS, Voxtral, VibeVoice, LFM2.5-Audio, Chatterbox) pass the same stage contract in tests but have no real-artifact worker evidence; Fish S2 intra-utterance streaming is covered by its own izwi-core ignored smoke test, not by worker evidence |
| Qwen3.5 chat (hybrid) | Excluded — hybrid reuse unproven (DS1.1 scope) | Excluded | Excluded |
| LFM2 chat (hybrid) | Excluded — hybrid reuse unproven | Excluded | Excluded |
| ASR / TTS / diarization / aligner families | Excluded — managed reuse is chat-task-gated | Excluded | Excluded |

This is an evidence statement, not a performance certificate: reuse on the
fixture lanes is counter-proven (counters + output equivalence), not
wall-clock-proven.

## Certified decode batch widths by family and backend (EXE-F7)

A chat family's continuous decode runs every row through one forward. The
widest batch it may run at on each backend is recorded in one table in code,
`crates/izwi-core/src/catalog/batch_width.rs`. Each loaded model clamps the
automatic engine width, and one derived from `IZWI_MAX_NUM_SEQS`, to its
family's width. Every other capability (TTS, ASR, realtime) stays at two on
CPU and Metal. An explicit `IZWI_MAX_BATCH_SIZE` applies as asked (up to 32
on any backend) to chat and speech models alike, above the certified width,
and the runtime logs a warning that the rows run uncertified.
The automatic engine width on CPU and Metal is the widest certified family
(eight). Staged transaction rows for TTS and ASR stay at two, and a chat
model stages one step at its own width.

A width is certified on CPU or Metal when both of these hold:

- Parity: greedy decode at that width picks the width-1 tokens, teacher-forced
  over 8 prompts. The width-1 top logits move by at most the tolerance.
- Throughput: aggregate tokens/s rise with the width in an interleaved
  in-process A/B. Each step of every width runs in turn, so machine load hits
  all widths alike.

The tests below are ignored and need the checkpoint.

| Family | CPU | Metal | CUDA | Evidence (M1 Pro, 2026-10-16) |
|---|---|---|---|---|
| LFM2 / LFM2.5 | 8 | 8 | 32 (unmeasured) | Parity `lfm25_decode_matches_across_batch_widths_on_{cpu,metal}`: widths 2, 4 and 8 match width 1 on every decisive step (339–340). Worst top-8 logit delta is 0 on the CPU and 0.0096 on Metal. Throughput `lfm25_batch_width_scaling_{cpu,metal}` at widths 1/2/4/8: Metal 133/153/226/258 and 91/136/210/258 tok/s; CPU 89/113/134/155 and 90/107/136/165 tok/s. Metal's small-batch mat-vec serves 2..=8 rows, which bounds the family at 8 |
| Qwen3.5 | 2 | 8 | 32 (unmeasured) | `qwen35_decode_width_parity_and_throughput_{cpu,metal}` (0.8B; 2B on Metal): every decisive step agrees with width 1 and the top-16 logits move by under 1e-4. Metal (2026-10-24, with the shared small-batch k-quant mat-vec on every Q4_K/Q5_K/Q6_K/Q8_0 projection and the tied head) 0.8B gives 105/180/251/285 tok/s at widths 1/2/4/8 (Candle per-row: 105/122/133/140); 2B gives 68/99/137/147 (60/69/73/77). On the CPU, 0.8B gives 21/22/25/26, then 11/16/15/15 on a repeat under load. That is no repeatable gain past two, so the CPU stays at 2 |
| Qwen3, Qwen3-MoE, Gemma3, Qwen3.6, Qwen3.8 | 2 | 2 | 32 (unmeasured) | No checkpoint on the measuring machine; the historical portable width |

CUDA keeps its automatic width of eight (VRAM-tiered admission) and a
ceiling of 32. The 32 is a policy constant (`catalog/batch_width.rs`), not a
kernel or metadata limit, and no CUDA width has been measured. An explicit
`IZWI_MAX_BATCH_SIZE` above a certified width is honoured up to 32 with a
startup warning.

Live check (Metal, LFM2.5-1.2B, release build). The sweep used 32 prompts of
256 words and 128 output tokens at rates 1, 2, 4 and inf, alternating runs on
one binary:

| Config | Output tok/s at rates 1/2/4/inf | TPOT p50 at rate inf |
|---|---|---|
| `IZWI_MAX_BATCH_SIZE=2` (before) | 82–107 | 74–80 ms |
| Automatic (8) | 118–157 at rates 2, 4 and inf | 46–51 ms |

These are three rounds. At rate 1 both configs run about 96–100 tok/s,
because the load is below what width two serves. Mean batch rows went from
2.0 to about 6.5, and TTFT p95 at rate inf went from 31–34 s to 22–24 s.
In one automatic round, live-memory admission refused 11 of 32 requests at
rate 4 with 503, because the shared machine was low on free memory. That
admission reserves a fixed amount per request and does not depend on width.
The other rounds had no failures.

## Scheduler step sizes by backend (SCH-F3)

| Backend | Token budget per step | Prefill chunk | Chunked prefill |
|---|---|---|---|
| CUDA | 2,048 | 512, kept while other rows decode | Always on (incremental chat) |
| CPU, Metal | 384 | 192 | Off by default: prompts prefill whole, so neither size bounds a step. On when `IZWI_MAX_NUM_SEQS` is above 1 (the multi-user profile), unless set explicitly |

`IZWI_MAX_TOKENS_PER_STEP` and `IZWI_CHUNKED_PREFILL_THRESHOLD` override both
sizes; `0` or unset takes the backend's. The CUDA sizes are unmeasured: no
NVIDIA device was available. The chunk no longer halves under decode load on
any backend. CPU and Metal keep the historical sizes, for two reasons.
Measured on Metal with LFM2.5 (1,000-word prompts at 2 req/s), the budget
does not bound a whole-prompt prefill, and decode needs at most 64 tokens a
step. Chunked prefill under concurrent long prompts used to fail decode rows
with "transaction N is active or has already reached a terminal state". That
was fixed in `9d9bae87`: the KV coordinator's terminal mark now covers only
transactions evicted from its exact window (Metal 9/24 → 24/24 requests,
CPU 9/16 → 16/16).

A family whose prompts must never prefill whole sets
`ModelFamily::requires_prefill_spans`; the scheduler then prefills it in
resumable spans whatever the chunked-prefill switch says
(`ExecutionProfile::prefill_spans_required`). Only Nemotron 3.5 Transcribe
sets it today: a one-hour recording is a prompt of about 45,000 tokens.

| Family | CPU | Metal | CUDA |
|---|---|---|---|
| Nemotron 3.5 Transcribe (ASR rows) | 512-token spans; experimental, uncertified (synthetic checkpoint only) | 512-token spans; experimental, uncertified (synthetic checkpoint only) | 2,048-token spans; experimental, uncertified (compile-checked only) |

`IZWI_NEMOTRON_TRANSCRIBE_PREFILL_SPAN` sets the span (16 to 16,384) on any
backend. No span size is measured: the model has not run with its real
weights. Its decode rows batch continuously at the engine's speech width, and
it reuses no prefixes. See [Nemotron 3.5 Transcribe](nemotron-transcribe.md).

## Mixed prefill+decode steps by family and backend (Phase 3a)

A chat family certified in `crates/izwi-core/src/catalog/mixed_step.rs`
runs prefill spans beside one-token decode rows in one model call and one
sampling readback (`chat.step.mixed`, one token budget per step). Everyone
else keeps the split prefill and decode stages. `IZWI_MIXED_STEP=0|1`
overrides the table (A/B runs and a kill switch), and `IZWI_MIXED_FORCE_SPLIT=1`
runs mixed batches through the split handlers (the parity oracle).

A cell is certified when the mixed forward matches the split calls (exact
where op shapes match, otherwise a tolerance with teacher-forced greedy
agreement) and the mixed step is a measured win: an interleaved in-process
A/B plus a live serving sweep with no tail regression.

| Family | CPU | Metal | CUDA | Evidence (M1 Pro, 2026-10-16) |
|---|---|---|---|---|
| LFM2 / LFM2.5 | Not certified | Not certified | Not run | Parity `lfm25_mixed_steps_match_split_calls_on_{cpu,metal}`: bit-exact on the CPU (logits, KV pages, ShortConv state); on Metal top-8 logits 0.33 (the Metal-vs-F32 reference error is also 0.33), KV 1.8%, every decisive step agrees. In-process A/B `lfm25_mixed_vs_split_step_timing_{cpu,metal}`: Metal mixed/split 0.90–0.96 for 8 decode rows beside spans of 64–376 tokens, 1.02–1.07 for 4 rows beside short spans; CPU 0.91–1.03 (prefill dominates, so no measurable gain). Live sweep below: no win on the ITL tail |
| Qwen3 | Not certified | Not certified | Not run | Trunk parity on the Qwen3-ASR-0.6B Q8 text model: bit-exact on the CPU, Metal F16 logits 0.055. No Qwen3 chat checkpoint to measure serving |
| Other chat families | Split | Split | Split | No mixed forward yet |

Live sweep (Metal, LFM2.5-1.2B, release build of this branch): 32 prompts of
500–1,500 words with 128 output tokens at rates 2, 4 and inf. `split` is the
default (chunked prefill off), `chunked` sets `IZWI_ENABLE_CHUNKED_PREFILL=1`,
`mixed` sets `IZWI_MIXED_STEP=1` (384-token budget), and `mixed128` adds
`IZWI_MAX_TOKENS_PER_STEP=128 IZWI_CHUNKED_PREFILL_THRESHOLD=120`. Clean
points only; round 2 lost requests to live-memory admission (503) on the
shared machine in every config:

| Config (rounds) | Output tok/s at 2/4/inf | TTFT p50 at 2/4/inf (s) | TPOT p50 (ms) | ITL p95 (ms) |
|---|---|---|---|---|
| split (2) | 80–89 | 15.5–17.4 / 21.6 / 22.0–22.2 | 78–90 | 74–95 |
| chunked (2) | 80–87 | 18.5 / 22.4 / 23.2–23.5 | 67–86 | 198–211 |
| mixed (2) | 73–92 | 17.0–21.4 / 21.3 / 24.4 | 72–92 | 317–371 |
| mixed128 (1) | 86–93 | 17.0 / 20.6 / 21.1 | 68–75 | 128–133 |

Every mixed step was one model call. A mixed step carries a prefill span
beside every decode row, so a 384-token span (about 0.35 s on Metal) lands
in every inter-token gap while prompts wait; the split default parks decode
only for the steps that prefill a whole prompt. A 128-token budget halves
the tail and beats chunked prefill on throughput, TTFT and ITL p95, but it
still trails the default's ITL p95 and has one round. LFM2 therefore stays
uncertified on CPU and Metal; CUDA is unvalidated (no NVIDIA device).

## Hierarchical KV offload by lane (DS4)

Offload is explicit opt-in (`host_kv_pool_budget_bytes` on the assignment or
`IZWI_KV_HOST_POOL_BUDGET_BYTES`; `IZWI_KV_HOST_OFFLOAD=0` is the kill
switch) and moves pages only between tiers of one worker process. Evidence
rig: `scripts/bench/run-ds4-offload-benchmark.sh` (shared workload, off/on
legs, sequential trailer, hard gates on completion, demotion, promotion, and
budget containment); acceptance test `ds4_host_offload.rs` (concurrent
shared-prefix sessions on an undersized arena, greedy replay byte-identical
to the cold run).

| Lane | Status | Evidence |
|---|---|---|
| CPU | Supported on the fixture lane — counter-proven (`benchmarks/manifests/ds4-cpu-summary.json`: on-leg demotions=10, promotions=8, host_pages=2 inside the 8 MiB budget; reuse preserved) | Not a production model, capacity, soak, or performance certificate |
| Metal | Supported on the fixture lane — counter-proven (`benchmarks/manifests/ds4-metal-summary.json`: on-leg demotions=7, promotions=3, host_pages=4; charged to the shared unified ledger, so the win is retention/admission headroom, never more memory) | Not a production model, capacity, soak, or performance certificate |
| CUDA | Not run — no hardware in this work session; the pool is designed as additional capacity across PCIe there | Recorded `not run`, never `passed` |

## Supervision (single node)

Each worker has its own supervision task, so one worker's slow load or crash
loop never stalls exit handling, restarts, diagnostics or autoscaling for the
others, and `max_parallel_model_loads` bounds concurrent loads. A ready
worker whose status stops answering (a wedged device or deadlock) is
restarted by the liveness probe. A quarantined worker is retried every
`restart.quarantine_retry_ms` and an operator clears it with SIGHUP or
`--clear-quarantine`. Operators pass `IZWI_*` switches and loader paths to
supervised workers through a validated `[workers.environment]` table.

| Scope | Status | Evidence |
|---|---|---|
| CPU lane (python fake workers serving the readiness contract) | Process-evidenced | `tests/supervision.rs`: a load that never finishes does not hold back the next worker or SIGUSR1 diagnostics (two load slots), one load slot serializes loads in config order, a worker whose status hangs is restarted into a new incarnation, and a quarantined worker is relaunched by the slow retry, by `--clear-quarantine` and by SIGHUP; launch and config unit tests for the environment table |
| Metal / CUDA lanes | Launch assignment process-evidenced (`multi_lane_launch.rs`); liveness and quarantine paths are backend-independent but were not run against a wedged real device | Recorded `not run` for real device faults |

## Coordinated rollout by scope (DS6)

One supervisor command (`--rollout-plan`) performs a deployment-generation
cutover. Blue-green (when both generations fit the node: plan validation
checks host memory, CPU threads, worker ids, binds and exclusive devices
together) launches the replacements canary-first, adds them to the shared
approvals file for a window in which the gateway cuts over without two or
zero eligible generations (DINV-07), then removes the old generation and
drains it. Recreate (`auto` picks it when the generations do not fit, as on
one Metal device or GPU) drains the old generation first and launches the
replacements, rolling back to the old generation if they fail. With a
`[gateway_ack]` section the rollout moves on only once every gateway's
`GET /internal/admin/approvals` reports the view adopted; without it, the
window and commit grace are timers sized from the declared refresh
interval. Aborts (readiness failure, replacement exit, SIGUSR2, shutdown, an
unacknowledged view) remove exactly the lines the rollout added under the
approvals file lock, so a peer's concurrent edit is kept. Design and
evidence: runbook section "Coordinated rollout (DS6)".

| Scope | Status | Evidence |
|---|---|---|
| One supervisor + gateways on the same host over one shared approvals file | Supported on the process-evidenced lane | Supervisor process tests (`tests/rollout.rs`): canary-failure abort, promotion through window → drain → commit with on-disk view assertions, replacement-exit abort, an abort that keeps a peer's edit made during the window, SIGKILL resume with fail-closed fresh start, status/abort command semantics, recreate on a shared bind (commit, and rollback when the replacement never gets ready), and acknowledgement against a fake gateway admin endpoint (commit and drain wait for it; an unacknowledged window aborts); gateway rig (`rollout_gateway.rs`, real gateway process): continuous traffic across window → abort-restore and window → commit with zero failed requests, and the admin route reports each view adopted and the cutover |
| Fleet-wide (multi-host) cutover | Not implemented in the supervisor — its views carry loopback worker endpoints. For fleets, roll generations with Kubernetes as the runbook's "Service managers and Kubernetes" section describes | Recorded `not implemented`, never `supported` |

`draining_old` is the point of no return (worker control-pipe EOF cannot be
un-sent); rollback after `committed` is a new rollout plan with a fresh
generation. Not a production soak or hardware-capacity certificate.

## Signal-driven worker autoscaling (DS7)

Optional and off by default (an `[autoscaling]` block enables it); fleets
should scale with Kubernetes and KEDA on the workers' metrics instead
(`config/serving/kubernetes/autoscaling.yaml`). The supervisor can scale one
deployment's worker count within explicit declared bounds: out on sustained
load (outstanding invocations), in after a stabilized idle window with
drain. It owns the v1 pinned approval lines of its autoscaled deployments in
the gateway's shared approvals file; the lines carry loopback endpoints, so
the gateways run on the same host. Scale-up and scale-down have separate
stabilization windows (scale-up 0 by default), decisions freeze while a
worker of the deployment is restarting, a crashed worker is never re-picked
as a standby, and config validation proves the whole declared replica set
fits the node. Design and evidence: runbook section "Signal-driven worker
autoscaling (DS7)".

| Scope | Status | Evidence |
|---|---|---|
| One supervisor, one autoscaled deployment, CPU lane | Supported on the process-evidenced lane (T38) | Policy unit tests (signals → decisions, per-direction windows, outstanding-load signal, lost-worker exclusion, bounds, view add/remove/reconciliation) in `crates/izwi-serving-supervisor/src/autoscale.rs`; process rig `tests/autoscale.rs` — sustained load scales 1→2 with the approvals line published only after readiness, the idle window scales 2→1 with unapprove → admission stop → drain observed and the core set untouched, the disabled configuration launches every declared worker and never touches the shared view, `--validate-only` reports the policy, and `--rollout-plan` is refused while autoscaling is on |
| One supervisor, multiple autoscaled deployments | Not run — the policy supports a map of deployments and the state machine is per-deployment, but the process rig exercises one deployment | Recorded `not run`, never `passed` |
| Metal / CUDA autoscaled lanes | Not run — scale-out on exclusive-device lanes requires distinct declared devices (config validation already enforces exclusivity); no multi-device autoscale rig exists | Recorded `not run`, never `passed` |

Not a production soak or hardware-capacity certificate; scale state is
in-memory by design (a supervisor restart returns to the declared min set).

## Worker protocol minor 4: one chat contract

A chat request runs on a worker exactly as on the local server. The worker
builds its engine request with the same izwi-core resolver (family sampling
profiles, seed, stop, top-k, penalties, logprobs, JSON mode, tools, and the D8
output budget and truncation), and the gateway sends each control only to a worker
that advertises its `WorkerFeature`; otherwise it returns a `400`
(`unsupported_parameter`), never dropping the control.

| Scope | Status | Evidence |
|---|---|---|
| Version negotiation and forward-compatible enums | Supported | Every worker-sent enum decodes a later variant as `Unknown`; the gateway sends `x-izwi-protocol-version` and the worker leaves out what an older gateway cannot decode; invocations carry the negotiated minor. JSON-literal old/new matrix in `crates/izwi-serving-protocol/tests/compatibility_matrix.rs` |
| Mixed fleets (minor 3 and minor 4 workers) | Supported | Controls route only to workers advertising them; plain requests still run on minor-3 workers at minor 3, which keep automatic truncation (`controls_route_only_to_workers_that_advertise_them`, which also covers a one-worker registry) |
| Rolling upgrade order | Either order from minor 4 on; minor-3 gateways decode minor-4 workers through the version header filter | Same tests |
| Local and gateway paths answer alike | Supported on CPU and Metal on the tiny LFM2 fixture; CUDA not run | `crates/izwi-serving-worker/tests/local_remote_parity.rs`: the local role's API and the gateway to a real worker process give identical greedy and seeded answers (content, logprobs, finish reason, token counts), stop strings, unset budgets, auto truncation and the D8 overflow `400`s, streamed and not (both admit before the stream starts); JSON mode reaches the engine on both, and a model failure mid-stream ends both streams with the same error event; both list exactly their one model on `/v1/models` |
| Tool calls through the gateway | Supported where the worker advertises `ChatTools` | The gateway sends `tools` to such workers and parses the model's tool-call markup from the worker's text with the local server's parser (`gateway_streams_worker_tool_markup_as_tool_calls`, scripted worker splitting the markup mid-tag); `local_remote_parity` shows a request with tools renders the same prompt and answer on both paths. The tiny fixture cannot emit a real call |

## API completeness surface (DS9)

OpenAI-shape API-completeness additions on the chat surface. All protocol
changes are additive (protocol minor 3): absent fields keep their previous
semantics, and older workers tolerate the new fields.

| Scope | Status | Evidence |
|---|---|---|
| `usage.cached_tokens` (DS9.1) | Supported | Scheduler-authoritative managed-prefix cursor through `ChatGeneration` into public `usage.prompt_tokens_details.cached_tokens` (chat) and `input_tokens_details.cached_tokens` (responses); scripted-runtime knob for tests; process test proves 64 cached tokens flow worker→public (`b2fb8415`) |
| `logprobs` / `top_logprobs` (DS9.3) | Supported | Per-token logprobs (raw-logit log_softmax) through all five families, non-streaming and streaming `choices[].logprobs`, and the gateway relay with a 65,536-entry cap; same-tokens property test pins that requesting logprobs never changes the token stream; T40 process evidence worker→public (`0f060c00`/`a3d34d83`/`77a07a1c`) |
| `response_format: json_object` on shared-sampler families (qwen3, gemma3, lfm2) (DS9.2) | Supported on the grammar-constrained lane | RFC 8259 grammar FSM masks at the ChatSampler seam (stop tokens stay sampleable, logprobs compose on the unmasked row), including continuous-batch rows; property test generates masked JSON across seeds and parses every document (`c1f4b1af`) |
| `response_format: json_object` on own-sampler families; `json_schema` everywhere | Documented 400, never silently ignored | Rejection is the recorded spike decision pending the schema→FSM follow-up (`c1f4b1af`) |
| Shared speculative MTP envelopes in continuous batches (DS9.4) | Opt-in (`IZWI_CUDA_MTP_IN_CONTINUOUS=auto`) on the CUDA+MTP lane: each row verifies with its own trunk pass, so it is off by default; perf claim hardware-gated | CPU fixture tests prove the two-row envelope commits exactly the solo sequence per row (greedy and sampled) with ragged exits and the opt-in gate both ways (`b6211f24`); the tokens/s-improvement / no-ITL-regression acceptance runs on CUDA via the qwen38 continuous-batching manifest's default (off) vs `IZWI_CUDA_MTP_IN_CONTINUOUS=auto` comparison using `summary.itl_ms` — recorded `not run` until hardware evidence exists |

The constrained-decoding spike deliberately ships the grammar machine per
state/key masking rather than a compressed-FSM grammar compiler; the
`json_schema` follow-up is the schema→FSM conversion noted above.

The only remotely advertised inference route in gateway mode is text-only
`POST /v1/chat/completions`, including its existing JSON and SSE response forms.
Every other route family remains explicitly local-only or absent.
Diarization and forced alignment have no worker task kind: they run only in
the full server (`--role local`), never on an `izwi-serving-*` worker (runbook,
"Route migration gates").

## Gateway failure semantics

One gateway makes no availability claim for gateway loss. With several fleet
gateways, losing one loses its in-flight streams and, when quotas are
partitioned, its quota slice until it restarts; the others keep serving.

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
  proven; it is not a crash-persistent or shared fleet authority. Fleet
  gateways share capacity-claim steering and (when partitioned) quota
  slicing through the coordination database; admission
  itself stays worker-authoritative and accepted-work ownership stays
  gateway-local. The T25 supervisor generation fence is process-evidenced
  (`tests/generation_fence_collision.rs`).
- Gateway mode intentionally owns no model runtime, accelerator,
  process-local session, or durable artifact provider, and no database unless
  scoped principal keys are configured (their store follows `IZWI_DB_PATH` /
  `IZWI_DATABASE_URL`) or the fleet profile's coordination database is set. Durable/local workflows do
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

Approve one exact cell explicitly. Unavailable lanes must remain `not run`
rather than `passed`.
