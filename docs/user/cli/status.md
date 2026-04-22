# izwi status

Show server health and status.

---

## Synopsis

```bash
izwi status [OPTIONS]
```

---

## Description

Displays the current state of the Izwi server, including health, loaded models, and runtime backend selection.

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

### Watch mode

```bash
izwi status --watch 2
```

Updates every 2 seconds. Press `Ctrl+C` to stop.

---

## Output

The status command shows:

- **Server health** — Running, stopped, or error
- **Loaded models** — Currently loaded models
- **Runtime backend** — Requested backend, selected backend, and fallback reason
- **Compiled backends** — CPU / Metal / CUDA support built into the server binary
- **Device summary** — Capability details reported by the selected device
- **Active requests** — Current request count

---

## See Also

- [`izwi serve`](./serve.md) — Start the server
- [`izwi models`](./models.md) — Model management
