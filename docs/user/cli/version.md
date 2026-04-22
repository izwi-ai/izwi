# izwi version

Show version information.

---

## Synopsis

```bash
izwi version [OPTIONS]
```

---

## Description

Displays the Izwi version and optionally detailed build information, including which backends were compiled into the CLI.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --full` | Show detailed version info | — |

---

## Examples

### Basic version

```bash
izwi version
```

Output:
```
izwi 0.1.0
```

### Full version info

```bash
izwi version --full
```

Output:
```
Version: 0.1.0

Build Info:
  Target:    darwin-aarch64
  Rust:      1.83.0

Compiled Backends:
  ✓ CPU
  ✓ Metal

Features:
  (none)
```

`Compiled Backends` reflects cargo features compiled into the binary. It does not mean the server is currently running on that backend; use [`izwi status --detailed`](./status.md) to verify runtime selection.

---

## See Also

- [`izwi status`](./status.md) — Server status
- [`izwi --version`](./index.md) — Quick version check
