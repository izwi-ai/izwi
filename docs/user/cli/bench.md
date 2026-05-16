# izwi bench

Run performance benchmarks.

---

## Synopsis

```bash
izwi bench <COMMAND>
```

---

## Subcommands

| Command | Description |
|---------|-------------|
| `chat` | Benchmark chat inference |
| `tts` | Benchmark TTS inference |
| `asr` | Benchmark ASR inference |
| `throughput` | Benchmark system throughput |
| `run` | Run a benchmark manifest |
| `compare` | Compare JSON reports and fail on regressions |

---

## izwi bench chat

Benchmark chat performance, including time-to-first-token.

```bash
izwi bench chat [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model to benchmark | `Qwen3.5-4B` |
| `-i, --iterations <N>` | Number of requests | `10` |
| `-p, --prompt <TEXT>` | User prompt to send | Default benchmark prompt |
| `--system <TEXT>` | Optional system prompt | — |
| `--max-tokens <N>` | Maximum completion tokens | `128` |
| `-c, --concurrent <N>` | Concurrent requests | `1` |
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench chat --model Qwen3.5-4B --iterations 8 --max-tokens 160 --warmup
```

---

## izwi bench tts

Benchmark text-to-speech performance.

```bash
izwi bench tts [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model to benchmark | `qwen3-tts-0.6b-base` |
| `-i, --iterations <N>` | Number of iterations | `10` |
| `-t, --text <TEXT>` | Text to synthesize | Default test text |
| `-c, --concurrent <N>` | Concurrent requests | `1` |
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench tts --model qwen3-tts-0.6b-base --iterations 20 --warmup
```

---

## izwi bench asr

Benchmark speech recognition performance.

```bash
izwi bench asr [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model <MODEL>` | Model to benchmark | `parakeet-tdt-0.6b-v3` |
| `-i, --iterations <N>` | Number of iterations | `10` |
| `-f, --file <PATH>` | Audio file to use | Built-in test audio |
| `-l, --language <LANG>` | Optional ASR language hint (e.g., `en`) | Auto-detect |
| `-c, --concurrent <N>` | Concurrent requests | `1` |
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench asr --model Whisper-Large-v3-Turbo --file test.wav --language en --iterations 20 --warmup
```

### Whisper ASR Performance Protocol

For CPU and Metal regressions, start a warmed server for the backend under
test, then run the smoke guardrail before recording a benchmark:

```bash
IZWI_BACKEND=metal \
IZWI_PRELOAD_MODELS=Whisper-Large-v3-Turbo \
IZWI_WARMUP_PRELOADED_MODELS=1 \
izwi serve --backend metal
```

```bash
IZWI_WHISPER_SMOKE_BACKEND=metal \
IZWI_WHISPER_SMOKE_SHORT_AUDIO=fixtures/short.wav \
IZWI_WHISPER_SMOKE_LONG_AUDIO=fixtures/long.wav \
IZWI_WHISPER_SMOKE_SILENCE_AUDIO=fixtures/silence.wav \
IZWI_WHISPER_SMOKE_NON_EN_AUDIO=fixtures/non-en.wav \
scripts/ci/whisper-smoke.sh
```

The smoke output must show Whisper diagnostics with
`device.model_dtype = "F32"`, `device.cuda_dtype_shim = false`, and
`device.whisper_impl = "local_whisper"` for CPU and Metal. CUDA runs keep the
CUDA-specific Whisper dtype policy.

Use JSON output for comparable benchmark artifacts:

```bash
izwi --output-format json bench asr \
  --model Whisper-Large-v3-Turbo \
  --file fixtures/short.wav \
  --language en \
  --iterations 10 \
  --warmup > whisper-current.json
```

```bash
izwi bench compare whisper-current.json whisper-baseline.json --tolerance-percent 10
```

Set `IZWI_WHISPER_PROFILE_SYNC=1` when investigating backend timing
attribution. Set `IZWI_WHISPER_DEVICE_GREEDY=0` only when isolating the scalar
device-greedy decode path from the incremental decoder cache.

For CUDA second-wave kernel benchmarks, build and compare Candle feature
combinations before changing CUDA defaults:

```bash
IZWI_CUDA_FEATURES=cuda scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,flash-attn scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,cudnn scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,cudnn,flash-attn scripts/ci/check-backend-truth.sh cargo-cuda
```

Run FlashAttention experiments with both the build feature and runtime opt-in:

```bash
IZWI_BACKEND=cuda \
IZWI_USE_FLASH_ATTENTION=1 \
IZWI_WHISPER_PROFILE_SYNC=1 \
izwi --output-format json bench asr \
  --model Whisper-Large-v3-Turbo \
  --file fixtures/short.wav \
  --language en \
  --iterations 10 \
  --warmup > whisper-cuda-flash.json
```

The CUDA FlashAttention path is Candle-backed and opportunistic: Whisper uses it
only for CUDA tensors when the build, dtype, head dimension, shape, and runtime
flag allow it. The benchmark telemetry should show fused-attention
attempt/success/fallback counters, and the transcript/WER must be compared
against the non-FlashAttention CUDA baseline. Only enable `cudnn` in builds
whose runtime image or host provides matching cuDNN libraries.

### Qwen-ASR CUDA Performance Protocol

Qwen-ASR CUDA kernel work follows the same guardrail rule as Whisper: prove
transcript quality and kernel telemetry before changing defaults. Start a server
with the backend and model under test, then run the Qwen smoke script before
recording benchmark artifacts:

```bash
IZWI_BACKEND=cuda \
IZWI_PRELOAD_MODELS=Qwen3-ASR-0.6B-GGUF \
IZWI_WARMUP_PRELOADED_MODELS=1 \
izwi serve --backend cuda
```

```bash
IZWI_QWEN_ASR_SMOKE_BACKEND=cuda \
IZWI_QWEN_ASR_SMOKE_SHORT_AUDIO=data/fox.wav \
IZWI_QWEN_ASR_SMOKE_LONG_AUDIO=data/diarization-2.mp3 \
IZWI_QWEN_ASR_SMOKE_SHORT_EXPECT="quick brown fox" \
IZWI_QWEN_ASR_SMOKE_LONG_EXPECTED_PREFIX="So human" \
IZWI_QWEN_ASR_SMOKE_REQUIRE_CHUNK_ATTENTION=1 \
scripts/ci/qwen-asr-smoke.sh
```

Run the same smoke script for CPU and Metal when changing shared Qwen code.
Those backends are protected behavior for CUDA second-wave work.

Build and compare the CUDA feature matrix before changing CUDA defaults:

```bash
IZWI_CUDA_FEATURES=cuda scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,flash-attn scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,cudnn scripts/ci/check-backend-truth.sh cargo-cuda
IZWI_CUDA_FEATURES=cuda,cudnn,flash-attn scripts/ci/check-backend-truth.sh cargo-cuda
```

Benchmark each candidate with JSON output. At minimum, compare the CUDA
baseline, dense decode, QMatMul, and FlashAttention runtime gates:

```bash
IZWI_BACKEND=cuda \
IZWI_QWEN3_DENSE_DECODE_ATTENTION=1 \
IZWI_QWEN3_ASR_GGUF_QMATMUL_TEXT=1 \
izwi --output-format json bench asr \
  --model Qwen3-ASR-0.6B-GGUF \
  --file data/diarization.wav \
  --iterations 10 \
  --warmup > qwen-asr-cuda-current.json
```

```bash
IZWI_BACKEND=cuda \
IZWI_USE_FLASH_ATTENTION=1 \
IZWI_QWEN3_DENSE_DECODE_ATTENTION=1 \
IZWI_QWEN3_ASR_GGUF_QMATMUL_TEXT=1 \
izwi --output-format json bench asr \
  --model Qwen3-ASR-0.6B-GGUF \
  --file data/diarization-2.mp3 \
  --iterations 5 \
  --warmup > qwen-asr-cuda-flash-long.json
```

Required review fields are the normalized transcript or WER, `RTF`,
`timings_ms.prefill`, `timings_ms.decode`, generated tokens, prompt/audio
tokens, fused attention attempt/success/fallback counts, chunk attention
fused/unfused/fallback counts, dense vs paged decode share, and Qwen
`execution` diagnostics for device, dtypes, dense decode, FlashAttention, and
text projection backend. FlashAttention
experiments require both the `flash-attn` build feature and
`IZWI_USE_FLASH_ATTENTION=1`; unsupported CUDA shapes must fall back through the
existing Candle path with telemetry rather than failing the request.
When validating audio chunk varlen FlashAttention specifically, also set
`IZWI_QWEN_ASR_SMOKE_REQUIRE_FUSED_CHUNKS=1` so the smoke script fails if the
runtime chunk-attention counters do not show fused spans.

---

## izwi bench throughput

Benchmark lightweight HTTP server overhead against `/livez`.

```bash
izwi bench throughput [OPTIONS]
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --duration <SECONDS>` | Test duration | `30` |
| `-c, --concurrent <N>` | Concurrent requests | `1` |

### Example

```bash
izwi bench throughput --duration 60 --concurrent 4
```

---

## izwi bench run

Run a TOML benchmark manifest with one or more benchmark cases.

```bash
izwi bench run benchmarks/local.toml
izwi bench run benchmarks/local.toml --artifact-dir benchmarks/results/run-001
```

### Manifest Format

```toml
server = "http://localhost:8080"

[[benchmarks]]
name = "chat-short-c1"
command = "chat"
model = "Qwen3.5-4B"
iterations = 10
concurrent = 1
warmup = true
prompt = "Summarize why batching helps transformer prefill."
max_tokens = 128

[[benchmarks]]
name = "asr-short-c2"
command = "asr"
model = "parakeet-tdt-0.6b-v3"
file = "data/test.wav"
iterations = 4
concurrent = 2
```

Supported `command` values are `chat`, `tts`, `asr`, and `throughput`. Relative ASR file paths resolve from the manifest directory.

Each benchmark can include a `[benchmarks.matrix]` table to generate a cartesian matrix from array values. Scalar values on the benchmark are used as defaults, and matrix values override them per generated case.

```toml
[[benchmarks]]
name = "chat-short"
command = "chat"
prompt = "Summarize why batching helps transformer prefill."
iterations = 10
warmup = true

[benchmarks.matrix]
model = ["Qwen3.5-4B", "Qwen3.5-8B"]
concurrent = [1, 2, 4]
max_tokens = [64, 128]
```

The example expands to 12 named cases such as `chat-short[model=Qwen3.5-4B,concurrent=1,max_tokens=64]`. Duplicate expanded case names are rejected so JSON reports can be compared safely by case identity.

When `--artifact-dir` is provided, the CLI writes:

- `report.json` — suite report with all case summaries and samples
- `manifest.toml` — copied manifest text
- `metadata.json` — CLI version, git SHA when available, platform, server, and run timestamps
- `observability.json` — before/after `/v1/health`, metrics JSON, and Prometheus telemetry snapshots

---

## izwi bench compare

Compare a current JSON report against a baseline JSON report. The command exits with a non-zero status when any comparable metric regresses beyond the tolerance.

```bash
izwi bench compare current.json baseline.json --tolerance-percent 5
```

Supported inputs are single reports from `--output-format json` and suite reports from `izwi bench run`. Lower-is-better checks include p95 latency, p95 TTFT, p95 end-to-end latency, and average RTF. Higher-is-better checks include request throughput, chat completion TPS, and TTS tokens/sec when present.

---

## Output

Benchmarks report:

- **TTFT** — Time to first streamed token for chat benchmarks
- **Latency** — Average, min, max, p50, p95, p99
- **Throughput** — Requests per second
- **Tokens/second** — For chat completion output and TTS server token generation when available
- **Real-time factor** — Audio duration vs processing time for TTS/ASR when available
- **Runtime telemetry snapshot** — Counter deltas for the run plus rolling runtime latency quantiles (explicitly labeled as rolling in CLI output)
- **Structured reports** — Use global `--output-format json` to emit a machine-readable report with run config, summary metrics, per-request samples, and runtime telemetry snapshots.

---

## Repeatable Kernel Performance Protocol

For regression tracking, use a fixed protocol so TTFT/TPS comparisons remain apples-to-apples:

1. Start from a warmed server and unchanged model weights.
2. Run both a short and long prompt profile with the same iterations and max tokens.
3. Capture both `izwi bench chat` output and `/internal/metrics/prometheus`.

### Recommended command set

```bash
# Short prompt profile (TTFT-sensitive)
izwi bench chat \
  --model Qwen3.5-4B \
  --iterations 20 \
  --max-tokens 128 \
  --warmup \
  --prompt "Summarize why batching helps transformer prefill in two concise bullet points."

# Long prompt profile (decode-path-sensitive)
izwi bench chat \
  --model Qwen3.5-4B \
  --iterations 20 \
  --max-tokens 128 \
  --warmup \
  --prompt "You are optimizing an inference runtime. Explain, in detail, how prefill and decode differ for attention kernels, why page-based KV caches help memory at long context, what tradeoffs they can introduce for throughput, and how to design a benchmark protocol that separates TTFT improvements from decode TPS improvements."

# Kernel-path counter snapshot
curl -sS http://127.0.0.1:11435/internal/metrics/prometheus | rg '^izwi_kernel_'
```

The `izwi bench chat` runtime delta now includes kernel-path counts for:
- prefill token-mode vs sequence-mode activity,
- dense vs paged decode attention routing,
- RoPE kernel vs manual path usage,
- fused attention attempts/success/fallback totals.

---

## See Also

- [`izwi status`](./status.md) — System status
