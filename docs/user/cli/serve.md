---
title: "izwi serve"
description: "Start the Izwi inference server and configure host, port, backends, logging, and runtime behavior."
icon: "server"
---
Start the Izwi inference server.

---

## Synopsis

```bash
izwi serve [OPTIONS]
```

---

## Description

Launches the local HTTP API server that powers Izwi.

The server also serves a local API reference:

- Scalar API reference: `http://<host>:<port>/docs`
- OpenAPI JSON: `http://<host>:<port>/openapi.json`

The local Scalar/OpenAPI reference covers the stable OpenAI-compatible contract,
the `/v1/responses` preview contract, readiness probes, and sidebar entries for
preview first-party, operator, and realtime route families. Detailed route
behavior is documented in the [API Reference](/api).

Resolution order for serve/runtime settings is:

1. CLI flags
2. Environment variables
3. `config.toml`
4. Built-in defaults

The resolved settings are passed through to the spawned `izwi-server` process, so `izwi serve` and `izwi-server` now run with the same runtime contract.

---

## Options

| Option | Description | Built-in default |
|--------|-------------|------------------|
| `--mode <MODE>` | Startup mode: `server`, `desktop`, `web` | `server` |
| `-H, --host <HOST>` | Host to bind to. A non-loopback host requires `IZWI_API_KEY` | `127.0.0.1` |
| `--allow-unauthenticated-network` | Start a non-loopback bind without an API key (unsafe) | Disabled |
| `-p, --port <PORT>` | Port to listen on | `8080` |
| `-m, --models-dir <PATH>` | Models directory | Platform default |
| `--max-batch-size <auto\|N>` | Physical tensor batch width (`auto` or a positive row count) | `auto` |
| `--backend <BACKEND>` | Backend preference: `auto`, `cpu`, `metal`, `cuda` | `auto` |
| `-t, --threads <N>` | Number of CPU threads | Auto (`available_parallelism`, capped at `8`) |
| `--max-concurrent <N>` | Max concurrent requests | `100` |
| `--timeout <SECONDS>` | Request timeout | `300` |
| `--log-level <LEVEL>` | Log level | `warn` |
| `--log-format <FORMAT>` | Log output format: `text` or `json` | `text` |
| `--cors` | Enable wildcard CORS responses | Disabled |
| `--no-ui` | Disable static web UI serving | UI enabled |

---

## Modes

### Server Mode

Starts only the HTTP server:

```bash
izwi serve
izwi serve --mode server
```

### Desktop Mode

Starts the server and opens the native desktop application:

```bash
izwi serve --mode desktop
```

### Web Mode

Starts the server and opens the browser.

```bash
izwi serve --mode web
```

When `--no-ui` is set, web mode opens `http://<host>:<port>/v1/ready` instead of the UI root.

---

## Examples

### Basic server

```bash
izwi serve
```

### Custom runtime settings

The Metal example requires Apple Silicon with macOS 15+.

```bash
izwi serve \
  --host 127.0.0.1 \
  --port 9000 \
  --backend metal \
  --max-batch-size 16 \
  --threads 6
```

### Access from other devices

Binding a network address requires an API key. Clients send it as
`Authorization: Bearer <key>` or `x-api-key: <key>`; the web UI asks for it
once.

```bash
export IZWI_API_KEY="$(openssl rand -hex 32)"
izwi serve --host 0.0.0.0
```

Without `IZWI_API_KEY` the server refuses to start on a non-loopback host.
`--allow-unauthenticated-network` overrides that and exposes model
management and history to anyone who can reach the port. See
[Security And CORS](/api#security-and-cors).

### Development browser access

```bash
izwi serve --cors --log-level debug
```

### API-only mode

```bash
izwi serve --mode web --no-ui
```

---

## Environment Variables

| Variable | Equivalent Option |
|----------|-------------------|
| `IZWI_HOST` | `--host` |
| `IZWI_PORT` | `--port` |
| `IZWI_API_KEY` | API key required for non-loopback binds (environment only) |
| `IZWI_ALLOW_UNAUTHENTICATED_NETWORK` | `--allow-unauthenticated-network` |
| `IZWI_MODELS_DIR` | `--models-dir` |
| `IZWI_BACKEND` | `--backend` |
| `IZWI_MAX_BATCH_SIZE` | `--max-batch-size` |
| `IZWI_MAX_TOKENS_PER_STEP` | Environment only: scheduler token budget per step (default `0`, the backend's: 2048 on CUDA, 384 on CPU/Metal) |
| `IZWI_CHUNKED_PREFILL_THRESHOLD` | No flag (also `runtime.chunked_prefill_threshold` in `config.toml`): prefill chunk in tokens when chunked prefill is on (default `0`, the backend's: 512 on CUDA, 192 on CPU/Metal) |
| `IZWI_NUM_THREADS` | `--threads` |
| `IZWI_MAX_CONCURRENT` | `--max-concurrent` |
| `IZWI_TIMEOUT` | `--timeout` |
| `IZWI_CORS` | `--cors` |
| `IZWI_CORS_ORIGINS` | CORS origin list (no flag; same as `server.cors_origins`) |
| `IZWI_NO_UI` | `--no-ui` |
| `IZWI_UI_DIR` | UI build directory (no flag; same as `ui.dir`) |
| `RUST_LOG` | `--log-level` |
| `IZWI_LOG_FORMAT` | `--log-format` |
| `IZWI_STREAM_SLOW_CONSUMER_TIMEOUT_MS` | Environment only: how long a streamed chat answer waits for a client that stopped reading before it fails (default `30000`). While the client lags, text is merged rather than dropped |
| `IZWI_SERVE_MODE` | `--mode` |
| `IZWI_ASR_REALTIME_NOISE_GATE` | Environment only: `0` stops silencing non-speech before it reaches Voxtral Realtime, which removes its 500 ms transcript delay but lets noise become text (default on) |
| `IZWI_VAD_SNR_MARGIN_DB` | Environment only: how far (dB) audio must rise above its own noise floor to count as speech, for the ASR no-speech check, realtime voice turns, `/transcription` and the Voxtral Realtime noise gate (default `6`). `0` turns the noise-floor check off |

Legacy aliases still resolve for one release cycle:

- `MAX_CONCURRENT_REQUESTS`
- `REQUEST_TIMEOUT_SECS`

---

## Configuration File

`izwi serve` also reads `config.toml`. Supported runtime keys:

```toml
[server]
host = "127.0.0.1"
port = 8080
cors = false
cors_origins = ["http://localhost:3000"]

[models]
dir = "/path/to/models"

[runtime]
backend = "auto"
max_batch_size = 8
threads = 8
max_concurrent = 100
timeout = 300

[ui]
enabled = true
dir = "ui/dist"
```

---

## Logging

Text logs are the default. Use JSON logs when a collector needs machine-readable application events:

```bash
izwi serve --log-format json
```

`--log-format json` controls Izwi's application log lines. Docker's `json-file` logging driver only frames container stdout/stderr and is separate from the application log format.

JSON logs include the normal tracing fields such as timestamp, level, target, and message. HTTP request events also include service, version, correlation ID, method, path, status, latency in milliseconds, and error fields when a request fails.

---

## Graceful Shutdown

Press `Ctrl+C` to gracefully shut down the server. Active requests finish before shutdown.

During startup, `izwi serve` waits for the readiness endpoint before opening desktop or web clients. Use `/readyz` for deployment readiness checks and `/livez` for cheap liveness checks; `/v1/health` remains the richer status payload used by `izwi status`.

---

## See Also

- [`izwi status`](/cli/status) — Check server health
- [`izwi config`](/cli/config) — Manage configuration
