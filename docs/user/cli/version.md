---
title: "izwi version"
description: "Show Izwi version, build information, and compiled backend support."
icon: "badge-info"
---
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
 ___ ____ ____
|_ _|_  /_  /  High-performance audio inference
 | | / / / /   Text-to-Speech & Speech-to-Text
|___/___/___|  Optimized for Apple Silicon & CUDA

Version: 0.1.0

For more information, visit: https://github.com/izwi-ai/izwi
```

For a single line, use `izwi --version`, which prints `izwi 0.1.0`.

### Full version info

```bash
izwi version --full
```

After the same banner, the output is:
```
Version: 0.1.0

Build Info:
  Target:    macos-aarch64
  Rust:      1.x.y

Compiled Backends:
  ✓ CPU
  ✓ Metal

License:
  Apache 2.0

For more information, visit: https://github.com/izwi-ai/izwi
```

`Compiled Backends` reflects cargo features compiled into the binary. It does
not mean the server is currently running on that backend; for example, a macOS
release can contain Metal code while selecting CPU on macOS 12-14. Use
[`izwi status --detailed`](/cli/status) to verify runtime selection.

---

## See Also

- [`izwi status`](/cli/status) — Server status
- [`izwi --version`](/cli) — Quick version check
