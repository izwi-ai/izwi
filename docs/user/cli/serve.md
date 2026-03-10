# izwi serve

Start the Izwi inference server.

---

## Synopsis

```bash
izwi serve [OPTIONS]
```

---

## Description

Launches the local HTTP API server that powers Izwi.

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
| `-H, --host <HOST>` | Host to bind to | `0.0.0.0` |
| `-p, --port <PORT>` | Port to listen on | `8080` |
| `-m, --models-dir <PATH>` | Models directory | Platform default |
| `--max-batch-size <N>` | Maximum batch size | `8` |
| `--backend <BACKEND>` | Backend preference: `auto`, `cpu`, `metal`, `cuda` | `auto` |
| `-t, --threads <N>` | Number of CPU threads | Auto (`available_parallelism`, capped at `8`) |
| `--max-concurrent <N>` | Max concurrent requests | `100` |
| `--timeout <SECONDS>` | Request timeout | `300` |
| `--log-level <LEVEL>` | Log level | `warn` |
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

When `--no-ui` is set, web mode opens `http://<host>:<port>/v1/health` instead of the UI root.

---

## Examples

### Basic server

```bash
izwi serve
```

### Custom runtime settings

```bash
izwi serve \
  --host 127.0.0.1 \
  --port 9000 \
  --backend metal \
  --max-batch-size 16 \
  --threads 6
```

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
| `IZWI_MODELS_DIR` | `--models-dir` |
| `IZWI_BACKEND` | `--backend` |
| `IZWI_MAX_BATCH_SIZE` | `--max-batch-size` |
| `IZWI_NUM_THREADS` | `--threads` |
| `IZWI_MAX_CONCURRENT` | `--max-concurrent` |
| `IZWI_TIMEOUT` | `--timeout` |
| `IZWI_CORS` | `--cors` |
| `IZWI_CORS_ORIGINS` | Config-only CORS origin list |
| `IZWI_NO_UI` | `--no-ui` |
| `IZWI_UI_DIR` | Config-only UI build directory |
| `RUST_LOG` | `--log-level` |
| `IZWI_SERVE_MODE` | `--mode` |

Legacy aliases still resolve for one release cycle:

- `MAX_CONCURRENT_REQUESTS`
- `REQUEST_TIMEOUT_SECS`

---

## Configuration File

`izwi serve` also reads `config.toml`. Supported runtime keys:

```toml
[server]
host = "0.0.0.0"
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

## Graceful Shutdown

Press `Ctrl+C` to gracefully shut down the server. Active requests finish before shutdown.

---

## See Also

- [`izwi status`](./status.md) — Check server health
- [`izwi config`](./config.md) — Manage configuration
