# Serving benchmark

`scripts/bench/run-serving-benchmark.py` measures chat serving the way
vLLM's `benchmark_serving.py` does. It sends streamed
`POST /v1/chat/completions` requests at a request rate (open loop) or under a
concurrency cap. It records the arrival time of every streamed delta and
reports latency distributions, throughput and goodput. Results go to a
versioned JSON file.

It needs only `python3` (3.8 or later, standard library). It works against
both roles: the local server (`izwi serve` / `izwi-server --role local`) and
the gateway (`izwi-gateway`, or `izwi-server --role gateway`, in front of
workers).

`izwi bench chat` remains the closed-loop, fixed-iteration tool. Its
per-request ITL is a mean, so it cannot show an inter-token tail.
`izwi bench throughput` was removed: it timed `GET /livez`, not inference.

## Quick start

Local role on this machine (loopback needs no key):

```bash
izwi serve --backend metal          # or cpu / cuda
python3 scripts/bench/run-serving-benchmark.py \
  --base-url http://127.0.0.1:8080 \
  --model LFM2.5-1.2B-Instruct-GGUF \
  --num-prompts 200 --input-len 256 --output-len 128 \
  --request-rate 1,2,4,8,inf \
  --slo-ttft-ms 1000 --slo-tpot-ms 50 \
  --metrics-url http://127.0.0.1:8080/v1/metrics/prometheus \
  --output benchmarks/results/serving/$(date -u +%Y%m%dT%H%M%SZ).json
```

Gateway (see the [runbook](PRODUCTION_SERVING_SINGLE_NODE_RUNBOOK.md) for
starting the gateway and workers). The inference key and the metrics key are
different secrets. Pass them through the environment so they stay out of the
process list:

```bash
export IZWI_BENCH_API_KEY=...            # the gateway's IZWI_GATEWAY_API_KEY
export IZWI_BENCH_METRICS_API_KEY=...    # the secret IZWI_GATEWAY_METRICS_API_KEY_REF names
python3 scripts/bench/run-serving-benchmark.py \
  --base-url http://127.0.0.1:8080 --model LFM2.5-1.2B-Instruct-GGUF \
  --backend metal \
  --request-rate 2,4,inf --num-prompts 200 \
  --metrics-url http://127.0.0.1:8080/internal/metrics/prometheus \
  --output gateway-sweep.json
```

The gateway has no `/v1/health`, so the script cannot read the worker's
backend. Pass `--backend` to label the results.

## Load shapes

| Flags | Load |
|---|---|
| `--request-rate 4` | Open loop, Poisson arrivals averaging 4 req/s. Each request is sent at its arrival time whether or not earlier ones have finished. |
| `--request-rate 1,2,4,8` | A sweep: one point per rate, run one after another. |
| `--request-rate inf` | All requests at once (a burst). |
| `--request-rate inf --max-concurrency 4` | Fixed concurrency: at most 4 in flight; each completion releases the next. |
| `--request-rate 4 --max-concurrency 8` | Open loop with a client-side cap. Requests over the cap wait in the client. Their latency is measured from the actual send, as in vLLM. |
| `--burstiness 0.5` / `2` | Gamma-distributed gaps with the same mean rate. Below 1 is burstier than Poisson; above 1 is more regular. |

Several `--max-concurrency` values are crossed with the rates. Each point is
named `rate=<r>[,concurrency=<c>]`.

Every point draws its own seeded prompts. A later point therefore does not
reuse the prefix cache that an earlier point filled. Warmup requests
(`--warmup N`) are unmeasured, run one at a time before the sweep, and use
separate prompts.

## Datasets

- `--dataset synthetic` (the default): random common-English-word prompts.
  `--input-len` sets the length in words and `--output-len` sets `max_tokens`.
  `--input-len-range r` and `--output-len-range r` draw lengths uniformly from
  `len*(1-r)` to `len*(1+r)`, as vLLM's random dataset does. A word is roughly
  one token. The token counts in the report come from the server's `usage`,
  not from the word counts.
- `--dataset FILE.jsonl`: one JSON object per line, either
  `{"prompt": "...", "output_len": n}` or
  `{"messages": [...], "max_tokens": n}`. A missing length falls back to
  `--output-len`.
- `--dataset sharegpt.json`: a ShareGPT JSON array. The first human turn is
  the prompt. The reply's length sizes `max_tokens`, estimated as characters/4
  because the client has no tokenizer.

Prompts are sampled with a seeded shuffle and cycled when the file has fewer
than `--num-prompts`. Decoding is greedy (`temperature 0`) unless
`--temperature` is set. With a temperature, request i uses seed `seed + i`.
izwi has no `ignore_eos`, so a request can stop before `max_tokens`. Check the
`output_tokens` distribution in the report.

## Reading the results

The summary printed per point and the `summary` object in the JSON contain:

| Field | Meaning |
|---|---|
| `completed`, `failed`, `aborted`, `errors` | `errors` counts failures by kind: `http_<status>` (for example `http_429` admission rejections), `stream_error:<code>` (an in-band OpenAI error event), `truncated` (the stream ended without `[DONE]`), `timeout` (over `--request-timeout`), and `client:<Exception>`. `aborted` = `timeout` + `truncated`. |
| `throughput_rps` | Completed requests per second of the point's duration (first send to last completion). |
| `output_throughput_tps`, `input_throughput_tps`, `total_token_throughput_tps` | Tokens per second, from the server's `usage` block. |
| `ttft_ms` | Send to the first delta that carries text (content, reasoning or tool call). |
| `itl_ms` | Every gap between successive text deltas of every request, pooled. This is the inter-token tail; a per-request mean would hide it. |
| `tpot_ms` | Per request: (e2e − TTFT) / (output tokens − 1). |
| `end_to_end_ms` | Send to the end of the stream. |
| `tokens_per_delta` | Output tokens / text deltas. Above 1 means the server coalesced tokens into one delta, so one ITL gap covers several tokens; read `tpot_ms` for per-token speed. |
| `goodput` | With `--slo-ttft-ms`, `--slo-tpot-ms` and/or `--slo-e2e-ms`: requests that completed within every given SLO, as req/s and as a fraction of all requests. |
| `warnings` | Percentiles that need more samples. p99 needs at least 100 samples and p95 at least 20; below that the reported value is at or near the maximum. Use `--num-prompts 200` or more for tail claims. |

Distributions report `count`, `mean`, `std`, `min`, `max` and `p50/p90/p95/p99`
(linear interpolation, numpy's default).

### Server-side and counted metrics

With `--metrics-url`, the script scrapes Prometheus text before and after each
point and stores the deltas under `server_metrics[<url>]`:

- `histograms`: every histogram series that moved, with count, mean and
  p50/p90/p99 from the bucket deltas (Prometheus `histogram_quantile`
  semantics, in the metric's own unit). This includes
  `izwi_request_ttft_seconds`, `izwi_request_queue_wait_seconds`,
  `izwi_engine_inter_token_latency_seconds`, and the gateway's
  `izwi_gateway_chat_*` families.
- `counters`: every `_total` counter that moved.
- `counted`: per-token and per-step figures to track across changes:

| Counted metric | Source |
|---|---|
| `device_readbacks_per_token` | Δ`izwi_kernel_host_read_ops_total` / output tokens. Output tokens are Δ`izwi_request_generation_tokens_total`, or the client's count when the server does not export it. Each read is one device-to-host synchronization. |
| `host_us_per_step` | Σ phases Δ`izwi_engine_step_host_seconds_sum` / Δ`_count{phase="prepare"}`: engine host time per step, with `step_host_us_by_phase` for `prepare` (scheduling, batch planning), `commit` (state commit per physical batch) and `deliver` (stream delivery). |
| `engine_steps`, `engine_steps_per_token` | Steps in the point; below 1 step per token means batching or multi-token quanta. |
| `batch_rows`, `batch_tokens` | `izwi_engine_batch_rows` / `_tokens` per dispatched physical batch (mean and quantiles). |
| `server_itl_ms` | The engine's per-token ITL histogram. A commit of k tokens after a gap g counts as k gaps of g/k, so it differs from the client's per-delta ITL by design. |
| `gateway_itl_ms`, `gateway_ttft_ms` | The gateway's view of the same streams. |

The engine families exist only on the local role
(`/v1/metrics/prometheus`). A worker's private
`/internal/v1/metrics/prometheus` exports fixed process counters, not engine
histograms. A gateway run therefore reports gateway-side histograms only.
Channel hops per token are not exported yet. Counters are process-wide: keep
other traffic off the server during a run, or the deltas include it.

### Metadata

`metadata` records:

- `model`, and `backend` with `backend_source` (`server_health` on the local
  role, `argument` from `--backend`);
- `git.sha` and `git.dirty` of the checkout that ran the script;
- `host`: OS, CPU model, core count, memory, and NVIDIA GPU name, driver and
  memory when `nvidia-smi` exists;
- `server.health`: the local role's `/v1/health` selection with
  `build_git_sha`, detected device, dtype policy, KV cache policy, chat
  concurrency policy and loaded models;
- `server.models`: `/v1/models`;
- `extra`: whatever `--metadata '{...}'` passed, for example the worker
  revision behind a gateway.

`config` holds every load and dataset setting, so a run can be repeated
exactly. `--save-samples` adds the per-request records (every ITL gap, token
counts, errors) to each point.

### Mixed prefill+decode A/B

To compare a family's mixed steps with the split stages, run the same sweep
against two servers of one binary: `IZWI_MIXED_STEP=0` and
`IZWI_MIXED_STEP=1`, alternating the order across rounds. Use prompts long
enough to need several steps (`--input-len 1000 --input-len-range 0.5`) at
rates that keep decodes running while prompts arrive. Read ITL p95 beside
throughput: a mixed step puts a prefill span into every inter-token gap,
so a throughput gain can come with a longer tail. The counted
`izwi_engine_model_calls_per_step` histogram (1 for a mixed step) and
`izwi_engine_mixed_rows{phase}` / `izwi_engine_mixed_tokens` show whether
the mixed stage served the run. Mixed steps run decode rows one token at a
time, so `tokens_per_delta` drops to 1 where the split path streams
multi-token quanta. The LFM2.5 results are in the support matrix.

### Device idle between steps

Every model call, KV page operation, model load and free runs on one thread
per device. `izwi_engine_device_idle_seconds{device}` records how long the
device waited between two model steps while the engine had work: the host
commit, delivery, prepare and dispatch between them.
`izwi_engine_device_ops_off_thread_total` counts device operations that ran
on any other thread; it must stay 0.

Measured on an M-series Mac (Metal, LFM2.5-1.2B-Instruct Q4_K_M, 32
requests at concurrency 8, 128 tokens in and out, machine shared with other
work): 628 gaps, mean 0.60 ms, 87% at or below 0.5 ms, against an ITL p50
of 31.8 ms (about 2% of a step). Per step, prepare took 0.33 ms, commit
0.11 ms and delivery 0.003 ms. Moving output assembly and delivery off the
gap (starting the next step after the KV commit) would save under 0.1 ms
of the 0.6 ms, so the step driver keeps prepare, execute and commit in
order. The ignored `step_host_overhead_bench` prints the same gap
(`device_idle`) for a mock model with a fixed device time.

### Long-form transcription beside chat

Long-form ASR (Whisper, Qwen3-ASR, Granite Speech, LFM2.5 Audio) runs one
audio window per engine dispatch (`asr.long_form.chunked` stages), not the
whole file in one dispatch. Between two windows the device runs other
work: control jobs (KV page operations, loads, frees) first, then model
dispatches by execution lane and workload-class weight
(`izwi_engine_device_queue_wait_seconds{device,class}`). A running chat
stream therefore waits at most one window per gap, and after each window a
ready chat prompt takes the next prefill slot before the transcription's
next window. VibeVoice ASR, offline Voxtral and Nemotron, and Parakeet
still transcribe a file in one dispatch.

`IZWI_LANE_STEP_DRIVERS=1` runs one step driver per loaded model instead of
one for the engine (default off). Lanes then prepare, commit and deliver
concurrently and an abort, load or purge waits only for its own lane. On
Metal and CUDA the physical admission still launches one dispatch at a
time, so the device time a long dispatch holds is bounded by the chunking
above, not by the lane drivers.

Measured on an M-series Mac (Metal, LFM2.5-1.2B-Instruct Q4_K_M chat at
`--request-rate 2 --num-prompts 16`, 128 tokens in and out, beside a
26-minute Whisper-Large-v3-Turbo transcription started 4 s before the chat
load; machine shared with other work, one run each):

| Build | Chat TTFT p50 / p95 | Chat ITL p50 / p95 / max | Transcription |
|---|---|---|---|
| Before (one dispatch per file) | 162.3 s / 167.5 s | 33.4 / 70.0 / 192 ms | 165 s (154 s alone) |
| After (one window per dispatch) | 12.9 s / 21.3 s | 27.5 / 1891 / 2365 ms | 155 s (148 s alone) |

Before, no chat stream ran while the transcription held the device: every
request waited for the whole file and then streamed normally, so its ITL
looks clean. After, streams run during the transcription and each window
(about 2-2.5 s on this machine) is one inter-token gap, which is the ITL
tail. Without the transcription both builds measured ITL p95 39-63 ms
across runs. Several requests failed with 503 in these runs because the
shared machine left little live memory (`insufficient live memory`), in
both builds.

## Audio load (ASR and TTS)

`scripts/bench/run-audio-load.py` is the open-loop load generator for
`/v1/audio/transcriptions` (`--task asr`) and `/v1/audio/speech`
(`--task tts`). It uses the same arrival process and metadata as the chat
benchmark and the same standard-library-only rule.

Every request has its own input, so batching sees mixed lengths. ASR clips
are cut from the `--audio` WAV files at seeded random offsets, with lengths
uniform in `--min-secs`..`--max-secs`. TTS inputs are seeded random sentences
of `--min-words`..`--max-words` words, each request with its own seed.

Before the load, every input runs alone once, and the first input runs a
second time to check that the model repeats itself. Under load each response
is compared with its solo output: ASR text must match exactly (a match that
ignores case and punctuation is reported too), and TTS audio must have the
same length and stay within `--tts-tolerance` of full scale. A batching bug
that changes one request's output therefore shows up as a parity failure,
not only as latency. A model that does not repeat its own solo output is
reported (`solo.repeatable: false`) and not held to parity. The samplers
treat seed 0 as "seed from the clock", so request seeds start at 1.

```bash
python3 scripts/bench/run-audio-load.py --task asr \
  --base-url http://127.0.0.1:8080 --model Qwen3-ASR-0.6B-GGUF \
  --audio data/test.wav --audio data/diarization.wav \
  --num-requests 16 --min-secs 2 --max-secs 20 --request-rate 1,inf \
  --output asr-load.json

python3 scripts/bench/run-audio-load.py --task tts \
  --base-url http://127.0.0.1:8080 --model Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --num-requests 12 --request-rate inf --temperature 0 \
  --tts-extra '{"subtalker_do_sample": false}' --require-parity
```

`--tts-extra` merges a JSON object into every speech request.
`--require-parity` exits 3 on any parity failure, or when the model does not
repeat itself. Without it, the run only reports.

Measured on an M1 Pro (Metal, one run each, at `--request-rate inf`):

| Model | Inputs | Audio s/s alone | Audio s/s under load | Parity |
|---|---|---|---|---|
| Qwen3-ASR-0.6B (GGUF) | 16 clips, 2-20 s | 18.2 | 21.3 | 16/16 exact |
| Granite Speech 5.0 2B (GGUF) | 12 clips, 2-20 s | 7.1 | 7.3 | 12/12 exact |
| Kokoro-82M | 16 inputs, 4-40 words | 6.5 | 6.5 | 16/16 exact |
| Qwen3-TTS 0.6B CustomVoice, greedy | 12 inputs, 4-30 words | 0.62 | 0.82 | 8/12 |

Kokoro and Granite 5 gain nothing under load. Granite 5 runs one request at
a time, and Kokoro's decoder batches only rows of equal duration, which
mixed inputs rarely are. Qwen3-TTS gains about a third, but four of twelve
greedy outputs differ from their solo audio: its batched path does not yet
reproduce single-request output.

## Regression check per backend

`--profile regression` is a small deterministic profile. It uses 24 seeded
synthetic prompts of 128 words and 32 output tokens, greedy decoding, and 2
warmup requests. It runs two points: `concurrency=1` (single-stream TTFT and
TPOT) and `concurrency=4` (batching). Explicit flags override the profile.

```bash
# Once per backend/hardware, on a quiet machine:
python3 scripts/bench/run-serving-benchmark.py --profile regression \
  --base-url http://127.0.0.1:8080 --model LFM2.5-1.2B-Instruct-GGUF \
  --output benchmarks/baselines/serving/m1pro-metal-lfm25.json

# After a change:
python3 scripts/bench/run-serving-benchmark.py --profile regression \
  --base-url http://127.0.0.1:8080 --model LFM2.5-1.2B-Instruct-GGUF \
  --baseline benchmarks/baselines/serving/m1pro-metal-lfm25.json \
  --output current.json

# Re-compare saved files without running anything:
python3 scripts/bench/run-serving-benchmark.py --current current.json \
  --baseline benchmarks/baselines/serving/m1pro-metal-lfm25.json --tolerance-percent 10
```

The comparison matches points by name. It checks request and output-token
throughput (higher is better) and TTFT p50/p95, TPOT p50, ITL p95 and e2e p95
(lower is better) against `--tolerance-percent` (default 10). It notes a
different model, backend, CPU, profile, dataset, lengths or seed. It is
report-only: the exit status is 0 unless `--fail-on-regression` is given, in
which case a regression exits 3. CI does not gate on timing, because shared
runners are too noisy. `izwi bench compare` reads the same files and checks
TTFT p95, e2e p95 and request throughput; it exits non-zero on a regression.

Wall-clock numbers move by 2× on a loaded machine. Treat a regression as a
reason to rerun on a quiet machine. Back performance claims with an
interleaved A/B or a counted metric (`device_readbacks_per_token`,
`host_us_per_step`), not with a single timing.

## Tests

`scripts/bench/test-serving-benchmark.py` covers the arrival process (Poisson
mean and spread), percentile math, the SSE parser (every split point, CR/LF
variants), goodput, Prometheus deltas and histogram quantiles, the result
schema, the comparison, and an end-to-end run against a local fake SSE
server. It runs in the `hygiene` lane of `scripts/ci/check-backend-truth.sh`.

`scripts/bench/test-audio-load.py` covers clip cutting, TTS input lengths,
WAV handling, the parity checks, the repeatability guard, and end-to-end runs
against a fake server that corrupts a response whenever requests overlap. It
runs in the same lane.
