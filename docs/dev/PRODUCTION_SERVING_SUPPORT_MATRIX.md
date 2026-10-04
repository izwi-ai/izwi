# Production-serving support matrix

Status: exact-profile support statement, not a production-readiness certificate

This matrix applies to Izwi's separated gateway/supervisor/worker architecture.
It does not change the support contract of the existing `--role local` desktop
and CLI workflows. A source path, parser test, mock worker, or compilation result
is never promoted to real hardware or performance evidence.

## Current profile cells

| Topology | Gateway | Worker/backend | Model/artifact evidence | Current status |
|---|---|---|---|---|
| One machine, one device | One hardware-independent gateway | One supervised CPU worker | Generated tiny GGUF for the already-supported `LFM2.5-1.2B-Instruct-GGUF` route completed JSON and SSE through a subprocess worker | Development vertical slice proven; not a production model, capacity, soak, or performance certificate |
| One machine, multiple devices | One gateway | Multiple independent mock workers | Deterministic chat replicas prove compatible routing, independent request identity, worker-authoritative capacity, retry fencing, and failure isolation | Mock transport evidence only; no physical multi-device claim |
| One machine, Apple silicon | One gateway | Metal worker (metal-feature build, device identity `metal:<registryID>` verified at startup) | **Real Metal inference executed**: 120 gateway-routed generations through a metal worker (debug build, tiny synthetic fixture, `benchmarks/manifests/ds07-metal-baseline.json`); `backend_parity.rs` proves greedy Metal outputs equal CPU outputs on the fixture, including the DS1.5 committed-prefix attach leg (`benchmarks/manifests/ds15-metal-summary.json`); supervisor launch + exact device identity process-evidenced (`multi_lane_launch.rs`) | Development Metal lane proven on the fixture; not a production model, capacity, soak, or performance certificate |
| One machine, NVIDIA | One gateway | CUDA-capable worker binary via `--cuda-worker-binary` | The supervisor accepts, validates, and launches CUDA assignments with exact declared UUID and device-visibility environment (process test `multi_lane_launch.rs`); no CUDA toolchain or NVIDIA device was available in this work session | Supervision code-complete; CUDA execution not run — no CUDA or multi-GPU support claim |
| Multiple machines | One gateway | Versioned node/worker-pinned HTTPS approvals | Fleet topology fails closed without mTLS client identity, HTTPS, and exact operator-pinned node/worker IDs; the bundled worker remains loopback-only and no proxy enforcement, separate-machine handshake, or artifact path was exercised | Not supported as an operational profile |
| Multiple gateways | Two or more gateways | Any worker fleet | Two real gateway processes over one shared coordination database prove T07 (atomic admission), T20 (partitioned quota not multiplied), P8.1 (shared approvals + monotonic observation registry), claim steering, gateway-crash + TTL recovery, and store-outage degradation (`tests/fleet_rig.rs`, SQLite and PostgreSQL lanes); capacity claims steer selection while the worker remains the atomic admission arbiter (ADR 0005) | Supported on the validated store lanes: SQLite (one host) and PostgreSQL (shared fleet); MySQL coordination SQL remains written but unvalidated; circuit state and accepted-work ownership stay per-gateway |

## Committed prefix reuse by model family and backend (DS1.6)

The catalog (`crates/izwi-core/src/catalog/prefix_reuse.rs`) records one cell
per model family per backend lane; serving surfaces default to catalog-auto,
which engages reuse only for cells with lane evidence. An explicit
`IZWI_ENABLE_PREFIX_CACHING=1` bypasses the table (salt required); an explicit
`0` is the kill switch. Auto degrades to Disabled with a recorded reason when
the runtime's page budget cannot fit a safe prefix reserve.

| Family | CPU | Metal | CUDA |
|---|---|---|---|
| Qwen3.8 chat (hybrid) | Supported — process parity (`backend_parity` prefix leg) | Supported — process parity (metal-feature leg) | Not enabled — no lane evidence |
| Qwen3 chat (dense) | Supported — DS1.2 fixture suite | Not enabled — contract declared, no lane parity | Not enabled — no lane evidence |
| Gemma3 chat (dense) | Supported — DS1.2 fixture suite | Not enabled — contract declared, no lane parity | Not enabled — no lane evidence |
| Voxtral (LM) | Not enabled — contract declared, no lane parity | Not enabled | Not enabled |
| Nemotron streaming ASR (`izwi-realtime-v1`) | Supported — real process evidence: the real worker binary with the real `Nemotron-3.5-ASR-Streaming-0.6B` artifact streamed `data/fox.wav` (36 frames, 24 kHz) through the WebSocket subprotocol → transcript "The quick brown fox jumps.", one Completed terminal, clean close (`real_cpu_realtime.rs`, ignored test, `IZWI_RT_EVIDENCE_BACKEND=cpu`) | Supported — same test with the metal-feature worker and device identity `metal:<registryID>` verified at startup (`IZWI_RT_EVIDENCE_BACKEND=metal`) | Not enabled — no lane evidence |
| Kokoro realtime TTS (`izwi-realtime-v1` TTS-stream) | Supported — real process evidence: the real worker binary with the real `Kokoro-82M` artifact synthesized the fox sentence over the subprotocol → 133,200 audio bytes (2.77 s at 24 kHz), non-silent, final-flagged terminal frame, one Completed terminal, clean close, HTTP-resolvable attempt (`real_cpu_realtime_tts.rs`, ignored test, `IZWI_RT_EVIDENCE_BACKEND=cpu`; ~41 s wall clock, RTF ≈ 15) | Supported — same test with the metal-feature worker and device identity verified at startup (`IZWI_RT_EVIDENCE_BACKEND=metal`; ~8 s wall clock) | Not enabled — no lane evidence. Other TTS families (Fish S2, Qwen3-TTS, Voxtral, VibeVoice, LFM2.5-Audio) pass the same stage contract in tests but have no real-artifact worker evidence; Fish S2 intra-utterance streaming is covered by its own izwi-core ignored smoke test, not by worker evidence |
| Qwen3.5 chat (hybrid) | Excluded — hybrid reuse unproven (DS1.1 scope) | Excluded | Excluded |
| LFM2 chat (hybrid) | Excluded — hybrid reuse unproven | Excluded | Excluded |
| ASR / TTS / diarization / aligner families | Excluded — managed reuse is chat-task-gated | Excluded | Excluded |

This is an evidence statement, not a performance certificate: reuse on the
fixture lanes is counter-proven (counters + output equivalence), not
wall-clock-proven.

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

## Coordinated blue-green rollout by scope (DS6)

One supervisor command (`--rollout-plan`) performs a deployment-generation
cutover: canary-first replacement launch, an atomic approvals view that
approves both generations for the soak window, a structurally
never-two-eligible / never-zero-eligible gateway cutover (DINV-07), then
drain of the old generation. Automatic aborts (canary/replacement readiness
failure, replacement exit during the window, SIGUSR2, shutdown) restore the
pre-rollout approvals byte-identically and leave the old generation serving
(ADR 0006). Design and evidence:
runbook section "Coordinated blue-green rollout".

| Scope | Status | Evidence |
|---|---|---|
| One supervisor + one gateway over one shared approvals file (single node) | Supported on the process-evidenced lane | Supervisor process tests (`tests/rollout.rs`): canary-failure abort with byte-identical approvals restore, promotion through window → drain → commit with on-disk view assertions, replacement-exit abort, SIGKILL resume with fail-closed fresh start, status/abort command semantics; gateway rig (real gateway process, mock workers on real TCP, T37 pattern): continuous traffic across window → abort-restore and window → commit with zero failed requests, and the draining predecessor receives nothing after cutover |
| Fleet-wide (multi-gateway) cutover | Not implemented — the coordinator writes one shared approvals file; each additional gateway adopts the same views, but no multi-gateway rollout rig or fleet-wide abort evidence exists | Recorded `not implemented`, never `supported` |

`draining_old` is the point of no return (worker control-pipe EOF cannot be
un-sent); rollback after `committed` is a new rollout plan with a fresh
generation. Not a production soak or hardware-capacity certificate.

## Signal-driven worker autoscaling (DS7)

The supervisor can scale one deployment's worker count within explicit
declared bounds: out on sustained queue depth, in after a stabilized idle
window with drain (ADR 0007). It is supervisor-managed — no Kubernetes, no
external scaler — and fleet-profile only: the supervisor owns the v1 pinned
approval lines of its autoscaled deployments in the gateway's shared
approvals file, and the gateway adopts the add/remove at runtime with no
gateway-side change. Scale decisions respect the node resource ledger
(declared host memory, CPU threads, device exclusivity; overcommitting
scale-ups are rejected with diagnostics, never launched), per-deployment
hysteresis, and drain-before-stop (never below the declared min set). Design
and evidence: runbook section "Signal-driven worker autoscaling (DS7)".

| Scope | Status | Evidence |
|---|---|---|
| One supervisor, one autoscaled deployment, CPU lane | Supported on the process-evidenced lane (T38) | Policy unit tests (signals → decisions, hysteresis, bounds, ledger overcommit, view add/remove/reconciliation) in `crates/izwi-serving-supervisor/src/autoscale.rs`; process rig `tests/autoscale.rs` — sustained queue depth scales 1→2 with the approvals line published only after readiness, the idle window scales 2→1 with unapprove → admission stop → drain observed and the core set untouched, the disabled configuration launches every declared worker and never touches the shared view, `--validate-only` reports the policy, and `--rollout-plan` is refused while autoscaling is on |
| One supervisor, multiple autoscaled deployments | Not run — the policy supports a map of deployments and the state machine is per-deployment, but the process rig exercises one deployment | Recorded `not run`, never `passed` |
| Metal / CUDA autoscaled lanes | Not run — scale-out on exclusive-device lanes requires distinct declared devices (config validation already enforces exclusivity); no multi-device autoscale rig exists | Recorded `not run`, never `passed` |

Not a production soak or hardware-capacity certificate; scale state is
in-memory by design (a supervisor restart returns to the declared min set).

## API completeness surface (DS9)

OpenAI-shape API-completeness additions on the chat surface. All protocol
changes are additive (protocol minor 3): absent fields keep their previous
semantics, and older workers tolerate the new fields.

| Scope | Status | Evidence |
|---|---|---|
| `usage.cached_tokens` (DS9.1) | Supported | Scheduler-authoritative managed-prefix cursor through `ChatGeneration` into public `usage.prompt_tokens_details.cached_tokens` (chat) and `input_tokens_details.cached_tokens` (responses); mock knob for tests; process test proves 64 cached tokens flow worker→public (`b2fb8415`) |
| `logprobs` / `top_logprobs` (DS9.3) | Supported | Per-token logprobs (raw-logit log_softmax) through all five families, non-streaming `choices[].logprobs`, streaming `delta.logprobs`, and the gateway relay with a 65,536-entry cap; same-tokens property test pins that requesting logprobs never changes the token stream; T40 process evidence worker→public (`0f060c00`/`a3d34d83`/`77a07a1c`) |
| `response_format: json_object` on shared-sampler families (qwen3, gemma3, lfm2) (DS9.2) | Supported on the grammar-constrained lane | RFC 8259 grammar FSM masks at the ChatSampler seam (stop tokens stay sampleable, logprobs compose on the unmasked row), including continuous-batch rows; property test generates masked JSON across seeds and parses every document (`c1f4b1af`) |
| `response_format: json_object` on own-sampler families; `json_schema` everywhere | Documented 400, never silently ignored | Rejection is the recorded spike decision pending the schema→FSM follow-up (`c1f4b1af`) |
| Shared speculative MTP envelopes in continuous batches (DS9.4) | Supported on the CUDA+MTP lane; perf claim hardware-gated | CPU fixture tests prove the two-row envelope commits exactly the solo sequence per row (greedy and sampled) with ragged exits and the opt-in gate both ways (`b6211f24`); the tokens/s-improvement / no-ITL-regression acceptance runs on CUDA via the qwen38 continuous-batching manifest's `IZWI_CUDA_MTP_IN_CONTINUOUS=off` vs default comparison using `summary.itl_ms` — recorded `not run` until hardware evidence exists |

The constrained-decoding spike deliberately ships the grammar machine per
state/key masking rather than a compressed-FSM grammar compiler; the
`json_schema` follow-up is the schema→FSM conversion noted above.

The only remotely advertised inference route in gateway mode is text-only
`POST /v1/chat/completions`, including its existing JSON and SSE response forms.
Every other route family remains explicitly local-only or absent.

## Single-gateway failure semantics

The current topology has one public gateway and makes no availability claim for
gateway loss.

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
  gateways share worker observations, capacity-claim steering, and (when
  partitioned) quota slicing through the coordination database; admission
  itself stays worker-authoritative and accepted-work ownership stays
  gateway-local. The T25 supervisor generation fence is process-evidenced
  (`tests/generation_fence_collision.rs`).
- Gateway mode intentionally owns no SQLite database, model runtime, accelerator,
  process-local session, or durable artifact provider. Durable/local workflows do
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
