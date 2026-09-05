# Qwen3.8 CUDA validation profiles

The shipping policy is **default-on portable CUDA optimization with explicit
configuration opt-outs**, as authorized on 2026-09-05. CUDA dispatch must resolve
by capability, dtype, shape, and resource budget. The NVIDIA L40S is the first
performance evidence target; it is not a production device-name condition.
CPU and Metal retain their existing routes. This policy supersedes the older
candidate-default-off protocol.

The implementation host has no NVIDIA device. Portable unit/reference/fixture
tests and driverless CUDA compilation do not establish GPU numerical parity,
throughput, memory peaks, or load-time improvement. Every hardware result is
bound to the exact source/build SHA, checkpoint revision, device UUID, driver,
compute/KV/provider selection and workload hash. An L40S result applies to that
profile only; default-on policy is not a claim of 40 t/s on every GPU.

## Configuration and rollback

Configuration precedence is defaults < TOML < environment < CLI. The typed
runtime policy lives under `[runtime.performance.cuda]` and
`[runtime.performance.loading]`. A subsystem `mode = "off"` dominates its
subordinate switches. Restart the server or reload the model after changing
policy; retain requested/resolved diagnostics and the reason for fallbacks.

Auto projection selection retains compact Q8 pending real-device crossover
evidence. Explicit `projection_backend = "native_fp8"` selects the implemented
software-decode FP8-weight/16-bit-activation (W8A16) provider on eligible paths.
It does not implement true FP8-activation Tensor Core W8A8. The provider name
alone must not be reported as proof of native FP8 arithmetic or a speedup.

| Subsystem | Automatic policy | Explicit opt-out |
|---|---|---|
| CUDA optimizations | `mode = "auto"` | `IZWI_CUDA_MODE=off` |
| Projection provider | `projection_backend = "auto"` | `IZWI_CUDA_PROJECTION_BACKEND=q8` |
| Packed projections | `packed_projections = "auto"` | `IZWI_CUDA_PACKED_PROJECTIONS=off` |
| Fused decode | `fused_decode = "auto"` | `IZWI_CUDA_FUSED_DECODE=off` |
| Device sampling | `device_sampling = "auto"` | `IZWI_CUDA_DEVICE_SAMPLING=off` |
| Decode graphs | `decode_graphs = "auto"` | `IZWI_CUDA_DECODE_GRAPHS=off` |
| MTP / sustained quantum | `mtp = "auto"`, `mtp_quantum = "auto"` | `IZWI_CUDA_MTP=off`, `IZWI_CUDA_MTP_QUANTUM=off` |
| Loading optimizations | `mode = "auto"` | `IZWI_LOADING_MODE=off` |
| Derived weights | `derived_weight_cache = "auto"` | `IZWI_LOADING_DERIVED_WEIGHT_CACHE=off` |
| Conversion workers | `parallel_conversion = "auto"` | `IZWI_LOADING_PARALLEL_CONVERSION=off` |
| Pinned upload | `pinned_uploads = "auto"` | `IZWI_LOADING_PINNED_UPLOADS=off` |

Loading also exposes `workers`, `max_staging_bytes`, `cache_max_bytes`, `cache_dir`
and `io_strategy` (`auto`, `mmap`, `sequential`). MTP begins at draft depth one;
`mtp_draft_tokens` supports 1–3 and `mtp_adaptive` controls adaptation. Existing
`IZWI_QWEN38_*` aliases remain supported, including explicit `0`; canonical
configuration takes precedence. BF16 KV remains a separate precision comparison,
not a consequence of FP8 checkpoint weights. An Auto request alone does not prove
that an optional kernel, graph, or provider executed.

Collect empty-config defaults, an explicit all-off baseline, each individual
optimization and opt-out, then combinations. The all-off server comparison uses
`IZWI_CUDA_MODE=off IZWI_LOADING_MODE=off`; MTP-depth comparisons instead hold
all other policy fixed and vary only MTP. Keep the exact environment/TOML/CLI
settings and server logs with each bundle.

## Timing and token contract

The user-visible metric is unchanged: committed completion tokens divided by
`izwi_generation_time_ms` (engine admission to response processing, including
initial queueing and prefill). Client end-to-end completion rate additionally
includes HTTP/stream overhead and uses its own denominator.

Relaxed OpenAI chat JSON and terminal SSE chunks carry `izwi_timing`:

- `queue_wait_ms`: engine admission to first scheduling;
- `prefill_ms`: attributed physical prefill service;
- `decode_ms`: sum of attributed physical decode batch durations; shared batches
  are attributed in full, so this is a service diagnostic;
- `decode_wall_ms`: an explicit monotonic timestamp at registration of the first
  entered decode dispatch through the last successful decode-token commit. It
  includes dispatch waiting, inter-quantum scheduling, retries and commit
  overhead, and excludes initial admission queue/prefill;
- `decode_tokens`: only tokens committed by decode quanta. A token sampled in
  prefill is part of completion usage but is excluded here. Draft proposals,
  rejected tokens, other requests and empty commits are never counted;
- `post_first_token_ms`: first committed token group to last committed token
  group. MTP can commit several tokens together; this is not per-token device
  latency and SSE chunk spacing must not be described as such;
- `ttft_ms`, `total_ms`, and prefill/decode step counts retain their distinct
  boundaries. Missing timing remains absent, never synthesized from total time.

Strict OpenAI mode omits Izwi extensions. Evidence requires relaxed mode. The
benchmark records actual `stop` versus `length` termination, raw `chat_timing`,
`server_request_tps` and `decode_wall_tps` summaries, including p10 rates and p95
latencies. Existing `generation_time_ms` and the displayed UI formula are preserved.
The direct Chat API stream now explicitly requests `stream_options.include_usage`.
Previously this path could fall back to `floor(text.length / 4)` when usage was
absent. Requesting true usage fixes token-count accuracy; it is not an inference
speedup. Baseline and candidate comparisons must both use actual committed token
counts, never compare the old character estimate with the new usage count and
credit that difference as performance. The original reported 16–18 t/s remains
unverified user evidence until the deployed SHA, exact prompt/settings and true
counts/denominator are collected.
Certificates rename the old misleading `decode_samples` to
`server_request_samples`; independent decode measurements are
`decode_wall_samples`. Legacy certificates lacking the new timing fail closed.

## L40S acceptance and reproduction

Use checkpoint `Qwen/Qwen3.8-27B-FP8` revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, compute capability 8.9 and the
48 GB profile in `benchmarks/manifests/qwen38-l40s-evidence.json`. Build release
CLI/server from the same checkout SHA with CUDA enabled. Record CPU/RAM/storage,
clocks/power/thermals, driver/toolkit/features, model revision and resolved policy.
Never mix profiler-overhead runs into acceptance timings.

Start one server with one model loaded. Explicitly set
`IZWI_ENABLE_PREFIX_CACHING=false` for primary cold-prefix acceptance. Every
benchmark request is a fresh stateless conversation; the runner rejects any
measured prefix-hit or reused-token increase. A fresh conversation alone does
not prove a cold prefix. Report warm-prefix reuse in a separately named profile.

The checked-in L40S thresholds are declared before measurement:

- At least ten independent runs of the exact user message
  **`Explain llm inference to me`**, with median server/UI request rate >=40 t/s.
- Ten 512-budget and ten 2,048-budget sustained runs, each at concurrency one,
  with median committed decode wall rate >=40 t/s. Their system instructions
  request long tutorials. Each sustained sample must actually commit at least
  384 or 1,536 decode tokens respectively. Early EOS is recorded and fails that
  sustained-length gate; the output budget is not a token-count measurement.
- Temperature zero and seed zero, fixed prompts/system/reasoning behavior and
  context within each comparison. The natural case may terminate early.
- Context targets 512/2K/8K/32K and concurrency 1/2/4/8 remain separate diagnostic
  cases. Aggregate/concurrent throughput cannot pass a single-sequence gate.
- Paired MTP also requires >=1.05× median client completion rate, <=1.05× p95
  TTFT regression and <=1.15× peak device memory for comparable cases. These are
  goals, not results. A missed gate remains an explicit limitation.

```bash
scripts/bench/run-qwen38-l40s-evidence.sh \
  --mtp-depth 1 --server http://127.0.0.1:8080 \
  --izwi-bin target/release/izwi --output target/qwen38-l40s-default
```

`--mtp-depth` asserts the expected loaded state; it does not configure the server.
Retain separate bundles for disabled MTP (`IZWI_CUDA_MTP=off`, depth 0) and
explicit depths 1/2/3 (`IZWI_CUDA_MTP=on`, `IZWI_CUDA_MTP_DRAFT_TOKENS=N`).
Also collect the unmodified empty-config policy separately. Then pair:

```bash
scripts/bench/certify-qwen38-mtp-evidence.sh \
  --baseline target/qwen38-mtp-disabled --depth-1 target/qwen38-mtp-depth-1 \
  --depth-2 target/qwen38-mtp-depth-2 --depth-3 target/qwen38-mtp-depth-3 \
  --output target/qwen38-mtp-paired
```

Four measured valid cells produce `runtime_validated`. Only candidates passing
both absolute single-sequence gates and relative MTP gates can produce
`performance_certified`. The certifier recomputes rates from token counts and
wall times and requires matched prompts, case counts, cache policy, model,
providers, SHA and physical hardware identity. A fixture test can exercise this
logic but cannot certify a GPU. Dry runs remain `implemented_unvalidated`.

Keep certificates, imported TOML, raw benchmark samples/telemetry, CLI metadata,
health, nvidia-smi memory/power/clock logs and matching server logs. For scheduler
grant evidence use `RUST_LOG=info,izwi_core::engine::scheduler=debug`. Retain
`Decode quantum granted` events with request_id, granted_tokens, preferred_tokens,
quantum_reason and sustained_decode_quantum. Aggregate per request to distinguish
model preference, token budget, peer fairness, soft deadline and scalar/workload
policy; process-global counters alone cannot prove a particular request's grants.

## Timed loading and first request

The family load manifest includes Qwen3.8. For a focused cell, use a text manifest
containing only `Qwen3.8-27B-FP8`. The load runner requires an empty exact-SHA CUDA
server before each load, checks actual CUDA residency, measures Ready and the
first real chat completion separately, then unloads. A failed first request or
unload fails the cell and stops later measurements.

```bash
scripts/bench/run-cuda-model-load-evidence.sh \
  --manifest benchmarks/manifests/qwen38-cuda-load.txt \
  --iterations 3 --cache-state reload --cache-provenance /path/to/provenance.json \
  --server-log /path/to/server.log --output target/qwen38-load-reload
```

`--cache-state` accepts `unknown`, `source-cold`, `source-warm`, `derived-cold`,
`derived-warm`, `reload`. Every non-unknown declaration requires an operator JSON
provenance file. Record how source/derived artifacts and OS cache were prepared,
storage type, source/cache digests and process identity. The runner never flushes
OS caches, never treats restart as a flush, and labels these as operator claims.
It independently records observed cache hits/misses as hit, partial hit, miss or
unobserved. Contradictions remain visible and disqualify the intended comparison.

Use `--prepare-run /absolute/executable` to prepare a cell and restart the server
externally before each run; it receives model and run index. Wait for health in
that hook and record a fresh process identity for fresh-process cells. Without
such external preparation, iterations are same-process unload/reloads. Do not
label them fresh-process measurements. `--first-request` accepts a chat JSON
body; Qwen3.8 defaults to the exact user prompt, greedy sampling and 8 output
tokens. Other model families need an appropriate first-request protocol.

The outer monotonic interval starts before load HTTP and ends at observed Ready
health (`load_ready_ms`). `first_request_ms` measures request execution/HTTP;
`first_request_ready_ms` spans load start to first request completion. The latter
includes deferred first-use work and all intervening inspection overhead.
`family_diagnostics.load_timing` supplies discovery, validation, conversion,
upload, cache counters and byte counters where measured. The current
`conversion_upload_timing_scope` is `q8-tiles-only`: dense/raw materialization is
not covered by those conversion/upload timings. Lifecycle weights/Ready wall
intervals still include that work. Unsupported/unmeasured
phases remain absent. Current cached-tile lookups rehash source bytes on every
lookup, preserving safety for mutable external source files. Consequently a
warm derived cache is not a zero-I/O reload; repeated source hashing remains a
loading limitation to measure. Do not claim immutable-source digest reuse or
zero source reads until that separate contract exists. Retained server logs contain `Model load reached physical
Ready` with artifacts_ms, admission_ms, weights_ms, upload_fence_ms,
state_allocation_ms, binding_publication_ms, preparation_ms and ready_ms. Phase
intervals may nest/overlap; do not sum them as an independent wall-clock total.

Collect at least three fresh-process runs per source-cold, source-warm,
derived-cold and derived-warm cell, then same-process reload. Keep download time
separate. Compare identical hardware/storage/process/cache states: >=20% intended
load improvement and <=5% p95 regression elsewhere are the acceptance goals;
2× conversion cold load and 5× conversion-dominated reload are stretch goals.
Absolute seconds require a measured baseline. The load runner is an evidence
collector with `performance_certified:false`, not a load-speed certifier.

## Cross-SM correctness and lifecycle scope

Use separate versioned hardware profiles for Ampere SM80/86, Ada SM89, Hopper
SM90 and actual Blackwell SM100/120 devices. Do not copy L40S's absolute 40 t/s
threshold into those profiles: declare relative improvement/no-regression gates
for their hardware. A shared compute_80 PTX floor, BF16 eligibility, native FP8
instructions and exact 128x128 block-scale support are independent contracts.
Unsupported specializations must resolve to a documented compact fallback.

On each available cell verify projection/recurrence/sampling numerics, all MTP
acceptance lengths, EOS/budget/cancellation, corrupt/stale caches and concurrent
writers, graph invalidation on allocation changes and reload, and bounded
RAM/pinned staging/VRAM. Verify actual route counters, not requested flags.
Run short/long context and the resource-fitted boundary, 20 sequential requests,
concurrency 1/2/4/8, cancellation during prefill/decode/load and stream disconnect.
Check managed-KV reservations/transactions and resident memory return to stable
post-request and post-unload levels; no hidden context reduction, offload,
precision change or attention truncation may satisfy the performance target.

Portable contract tests, driverless compile/link, real-device correctness and
exact-SHA performance certification are separate evidence tiers. Default-on
policy does not merge those tiers or erase unmeasured hardware cells.

## Local implementation verification (2026-09-05)

The macOS development checks cover portable numerical and state tests, the
core/CLI/server regression suites, desktop configuration startup, workspace
Clippy and all-target checks, UI timing tests, and evidence-runner fixtures.
The combined macOS build enables Accelerate; its existing predictor batch
projection now compacts the selected token rows, and the RoPE reference test
allows 1e-6 F32 error between scalar and vector trig implementations.

CUDA-conditional Rust types and Clippy were also checked in an isolated temporary
source copy with declaration-only PTX placeholders. No placeholder enters the
production tree. This catches Rust API errors but **does not compile the CUDA
C++ sources, link a CUDA binary, or execute a kernel**. Run both the real
`cargo-cuda-compile` and `cargo-cuda-device` CI profiles on the appropriate
host before treating this implementation as CUDA-certified. The latter requires
`IZWI_REQUIRE_CUDA_TEST_DEVICE=1` for the Qwen3.8 kernel, model and physical-state
suites and must not silently pass without a usable device.

No 27B checkpoint load, Nsight capture, L40S 40 t/s acceptance result, or
cross-SM performance result was obtained here. The commands and manifests above
define those remaining release gates; they are not substituted by local tests.
