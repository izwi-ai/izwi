---
title: "izwi list"
description: "List available and downloaded Izwi models with status and storage details."
icon: "list"
---
# izwi list

List available models.

---

## Synopsis

```bash
izwi list [OPTIONS]
```

---

## Description

Shows all available models, including both downloaded models and models available for download.

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-l, --local` | Show only downloaded models | — |
| `-d, --detailed` | Show detailed information | — |

---

## Examples

### List all models

```bash
izwi list
```

### List downloaded models only

```bash
izwi list --local
```

### Detailed view

```bash
izwi list --detailed
```

### JSON output

```bash
izwi list --output-format json
```

---

## Output

The list shows:

| Column | Description |
|--------|-------------|
| **Model** | Model variant name |
| **Status** | Downloaded, Ready, Not Downloaded |
| **Size** | Download size |
| **Progress** | Download progress percentage (`--detailed`) |
| **Path** | Local model directory (`--detailed`) |

---

## Status Values

| Status | Description |
|--------|-------------|
| `ready` | Loaded and ready for inference |
| `downloaded` | On disk but not loaded |
| `downloading` | Currently downloading |
| `not_downloaded` | Available but not downloaded |

---

## See Also

- [`izwi pull`](./pull.md) — Download models
- [`izwi rm`](./rm.md) — Remove models
- [`izwi models`](./models.md) — Model management
