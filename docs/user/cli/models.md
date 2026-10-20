---
title: "izwi models"
description: "Use model management subcommands to list, inspect, load, unload, and track model progress."
icon: "database"
---
Model management commands.

---

## Synopsis

```bash
izwi models <COMMAND>
```

---

## Subcommands

| Command | Description |
|---------|-------------|
| `list` | List available models |
| `info` | Show model information |
| `load` | Load a model into memory and wait until it is ready |
| `unload` | Unload a model from memory |
| `progress` | Show download progress |

---

## izwi models list

List available models.

```bash
izwi models list
izwi models list --local
izwi models list --detailed
```

Same as [`izwi list`](/cli/list).

---

## izwi models info

Show detailed information about a model.

```bash
izwi models info <MODEL>
izwi models info Qwen3-TTS-12Hz-0.6B-Base
izwi models info Qwen3-TTS-12Hz-0.6B-Base --json
```

### Options

| Option | Description |
|--------|-------------|
| `--json` | Output raw JSON |

---

## izwi models load

Load a model into memory and wait until it is ready. The model must be downloaded first: `izwi models load` refuses a model that is not downloaded (run `izwi pull <MODEL>`).

```bash
izwi models load <MODEL>
izwi models load Qwen3-TTS-12Hz-0.6B-Base
izwi models load Qwen3-TTS-12Hz-0.6B-Base --timeout 1200
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--timeout <SECS>` | Seconds to wait for the load | `600` |

---

## izwi models unload

Unload a model from memory.

```bash
izwi models unload <MODEL>
izwi models unload Qwen3-TTS-12Hz-0.6B-Base
izwi models unload all --yes
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model variant to unload, or `all` |

### Options

| Option | Description |
|--------|-------------|
| `-y, --yes` | Unload without confirmation |

---

## izwi models progress

Show download progress for a model. The model argument is required.

```bash
izwi models progress <MODEL>
izwi models progress Qwen3-TTS-12Hz-0.6B-Base
```

---

## See Also

- [`izwi list`](/cli/list)
- [`izwi pull`](/cli/pull)
- [`izwi rm`](/cli/rm)
