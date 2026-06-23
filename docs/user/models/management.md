---
title: "Model Management"
description: "Download, load, unload, delete, filter, and inspect Izwi models from the UI, CLI, and local API."
sidebarTitle: "Management"
icon: "hard-drive"
---
# Model Management

The **Models** workspace is the main place to manage local model files and
runtime readiness.

---

## Web UI

Open **Models** from the bottom of the sidebar.

The page shows:

- Total visible models
- Downloaded and loaded counts
- Total local model storage used
- Provider-grouped model cards
- Search, status filters, and category filters

Status filters:

| Filter | Meaning |
|--------|---------|
| All | Show every enabled catalog model. |
| Loaded | Show models with `ready` status. |
| Downloaded | Show models downloaded to disk but not loaded. |
| Not downloaded | Show models available for download. |

Category filters:

- Text to Speech
- Transcription
- Chat

---

## Model Statuses

| Status | Meaning | Typical action |
|--------|---------|----------------|
| `not_downloaded` | Available in the catalog, not on disk | Download |
| `downloading` | Download in progress | Wait or cancel |
| `downloaded` | Files are on disk, model is not in memory | Load or delete |
| `loading` | Runtime is loading the model | Wait |
| `ready` | Model is loaded and available for inference | Use or unload |
| `error` | Download/load failed | Refresh, retry, or inspect logs |

---

## Actions

| Action | Where | Notes |
|--------|-------|-------|
| Refresh | Models page header | Re-reads catalog and local status. |
| Download | Model card | Downloads model files into the configured models directory. |
| Cancel | Downloading model card | Cancels an active download when the runtime supports cancellation. |
| Load | Downloaded model card | Loads the model into memory. |
| Unload | Ready model card | Frees runtime memory while keeping files on disk. |
| Delete | Downloaded or ready model card | Removes local model files after confirmation. |

Gemma models may show **Manual DL** because they require manual Hugging Face
access setup. See [Manual Model Downloads](/models/manual-download).

---

## CLI

```bash
# List catalog models
izwi list

# List downloaded models only
izwi list --local

# Download
izwi pull Qwen3-TTS-12Hz-0.6B-Base

# Load and unload
izwi models load Qwen3-TTS-12Hz-0.6B-Base
izwi models unload Qwen3-TTS-12Hz-0.6B-Base

# Delete files
izwi rm Qwen3-TTS-12Hz-0.6B-Base

# Runtime status
izwi status --detailed
```

---

## API

Model management routes live under `/v1/admin/models`:

| Route | Purpose |
|-------|---------|
| `GET /v1/admin/models` | List model catalog and local status. |
| `POST /v1/admin/models/{variant}/download` | Start a download. |
| `GET /v1/admin/models/{variant}/download/progress` | Read download progress. |
| `POST /v1/admin/models/{variant}/download/cancel` | Cancel a download. |
| `POST /v1/admin/models/{variant}/load` | Load into memory. |
| `POST /v1/admin/models/{variant}/unload` | Unload from memory. |
| `DELETE /v1/admin/models/{variant}` | Delete local files. |

See the [API Reference](/api#admin-model-management) for response shapes and
capability fields.

---

## See Also

- [Models](/models)
- [Manual Model Downloads](/models/manual-download)
- [CLI Model Commands](/cli/models)
