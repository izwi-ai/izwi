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
| `--warmup` | Enable warmup iteration | — |

### Example

```bash
izwi bench asr --model Whisper-Large-v3-Turbo --file test.wav --language en --iterations 20 --warmup
```

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

## Output

Benchmarks report:

- **TTFT** — Time to first streamed token for chat benchmarks
- **Latency** — Average, min, max, p50, p95, p99
- **Throughput** — Requests per second
- **Tokens/second** — For TTS benchmarks
- **Real-time factor** — Audio duration vs processing time
- **Runtime telemetry snapshot** — Counter deltas for the run plus rolling runtime latency quantiles (explicitly labeled as rolling in CLI output)

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
