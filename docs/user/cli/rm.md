---
title: "izwi rm"
description: "Remove a downloaded Izwi model from local storage."
icon: "trash-2"
---
Remove a downloaded model.

---

## Synopsis

```bash
izwi rm <MODEL> [OPTIONS]
```

---

## Description

Deletes a downloaded model from local storage, freeing disk space.

---

## Arguments

| Argument | Description |
|----------|-------------|
| `<MODEL>` | Model variant to remove |

---

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-y, --yes` | Remove without confirmation | — |

---

## Examples

### Remove a model

```bash
izwi rm qwen3-tts-0.6b-base
```

### Skip confirmation

```bash
izwi rm qwen3-tts-0.6b-base --yes
```

---

## Notes

- The model will be unloaded from memory if currently loaded
- The model can be re-downloaded with `izwi pull`
- This operation cannot be undone

---

## See Also

- [`izwi list`](/cli/list) — List models
- [`izwi pull`](/cli/pull) — Download models
