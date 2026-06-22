---
title: "Manual Model Downloads"
description: "Manually download gated or externally hosted model files and place them in the Izwi model cache."
sidebarTitle: "Manual Downloads"
icon: "folder-down"
---
# Manual Model Downloads

Some models require manual download from Hugging Face due to licensing requirements or access restrictions. This guide covers the general process.

---

## When Manual Download is Required

You'll need to manually download models when:

- The model requires accepting a license agreement
- The model is gated (requires Hugging Face authentication)
- You want to use a custom or fine-tuned model
- Automatic download fails due to network issues

---

## Prerequisites

### Install Hugging Face CLI

Choose one method:

**Option 1: Using pipx (Recommended)**

```bash
pipx install huggingface_hub
```

**Option 2: Using pip**

```bash
python3 -m pip install --upgrade "huggingface_hub[cli]"
```

### Authenticate with Hugging Face

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co)
2. Generate an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
3. Log in via CLI:

```bash
huggingface-cli login
```

Enter your token when prompted.

---

## Download Process

### Step 1: Find the Model Directory

Izwi stores models in a specific location:

| Platform | Location |
|----------|----------|
| **macOS** | `~/Library/Application Support/izwi/models/` |
| **Linux** | `~/.local/share/izwi/models/` |
| **Windows** | `%APPDATA%\izwi\models\` |

### Step 2: Download the Model

Use the Hugging Face CLI to download:

```bash
huggingface-cli download <repo-id> \
  --repo-type model \
  --local-dir "<izwi-models-path>/<model-name>"
```

**Example for macOS:**

```bash
huggingface-cli download google/gemma-3-1b-it \
  --repo-type model \
  --local-dir "$HOME/Library/Application Support/izwi/models/Gemma-3-1b-it"
```

**Example for Linux:**

```bash
huggingface-cli download google/gemma-3-1b-it \
  --repo-type model \
  --local-dir "$HOME/.local/share/izwi/models/Gemma-3-1b-it"
```

### Step 3: Verify the Download

```bash
izwi list --local
```

The model should appear in the list.

### Step 4: Load the Model

```bash
izwi models load <model-name>
```

---

## Downloading Specific Files

If you only need certain files (e.g., to save space):

```bash
huggingface-cli download <repo-id> \
  --include "*.safetensors" "*.json" \
  --local-dir "<path>"
```

---

## Resuming Interrupted Downloads

The Hugging Face CLI automatically resumes interrupted downloads. Just run the same command again.

---

## Using a Custom Cache Directory

By default, Hugging Face caches downloads in `~/.cache/huggingface/`. To use a different location:

```bash
export HF_HOME=/path/to/cache
huggingface-cli download <repo-id> --local-dir "<path>"
```

---

## Common Models Requiring Manual Download

| Model | Repository | Notes |
|-------|------------|-------|
| Gemma 3 1B | `google/gemma-3-1b-it` | Requires license acceptance |
| Llama 3 | `meta-llama/Llama-3-*` | Requires license acceptance |

See specific guides:
- [Gemma 3 1B Download](/models/manual-gemma-3-1b-download)

---

## Troubleshooting

### "Access denied" or "401 Unauthorized"

1. Ensure you're logged in:
   ```bash
   huggingface-cli whoami
   ```

2. Check that you've accepted the model's license on Hugging Face

3. Verify your token has read permissions

### Download is very slow

Try using `hf_transfer` for faster downloads:

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download <repo-id> --local-dir "<path>"
```

### Model not detected by Izwi

1. Verify the model is in the correct directory
2. Check the folder name matches what Izwi expects
3. Restart the Izwi server:
   ```bash
   izwi serve
   ```

### Disk space issues

Check available space before downloading:

```bash
df -h
```

Large models can be 10+ GB. Ensure you have enough free space.

---

## Next Steps

- [Models Overview](/models)
- [Troubleshooting](/troubleshooting)
