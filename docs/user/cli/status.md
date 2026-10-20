---
title: "izwi status"
description: "Check Izwi server health, model readiness, runtime status, and detailed diagnostics."
icon: "activity"
---
Show server health and status.

---

## Synopsis

```bash
izwi status [OPTIONS]
```

---

## Description

Displays the current state of the Izwi server, including health, loaded models, and runtime backend selection. This command reads `/v1/health`, the rich runtime status payload.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --detailed` | Show detailed metrics | — |
| `-w, --watch <SECONDS>` | Continuous updates | — |

---

## Examples

### Basic status

```bash
izwi status
```

`izwi status` exits with status 1 when the server is unreachable or unhealthy, so scripts can use it as a check.

### Detailed metrics

```bash
izwi status --detailed
```

This reads the server health payload and reports:

- the requested backend (`auto`, `metal`, `cuda`, or `cpu`)
- whether that requested backend is currently available
- the backend the server actually selected
- whether the selection came from the request or a fallback path
- which backends were compiled into the running binary
- detected device capabilities such as BF16, unified memory, batch size, and memory when available

Deployment healthchecks should use `/readyz` for readiness and `/livez` for liveness. `izwi status` intentionally stays on `/v1/health` so it can report backend and model details.

### Watch mode

```bash
izwi status --watch 2
```

Updates every 2 seconds. Press `Ctrl+C` to stop.

---

## Output

The status command shows:

- **Server health** — Healthy, or Unhealthy with the HTTP status; an unreachable server is an error
- **Loaded models** — Currently loaded models
- **Runtime backend** — Requested backend, selected backend, and fallback reason
- **Compiled backends** — CPU / Metal / CUDA support built into the server binary
- **Device summary** — Capability details reported by the selected device

---

## See Also

- [`izwi serve`](/cli/serve) — Start the server
- [`izwi models`](/cli/models) — Model management
