---
title: "Manual Model Downloads"
description: "Manually download gated or externally hosted model files and place them in the Izwi model cache."
sidebarTitle: "Manual Downloads"
icon: "folder-down"
---
Some models require manual download from Hugging Face due to licensing requirements or access restrictions. This guide covers the general process.

---

## When Manual Download is Required

You'll need to manually download models when:

- The model requires accepting a license agreement
- The model is gated (requires Hugging Face authentication)
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
| **Windows** | `%LOCALAPPDATA%\izwi\models\` |

### Step 2: Download the Model

Use the Hugging Face CLI to download:

```bash
huggingface-cli download <repo-id> \
  --repo-type model \
  --local-dir "<izwi-models-path>/<model-name>"
```

`<model-name>` is the exact model ID from `izwi list` (for example `Kokoro-82M`).

**Example for macOS:**

```bash
huggingface-cli download <repo-id> \
  --repo-type model \
  --local-dir "$HOME/Library/Application Support/izwi/models/<model-name>"
```

**Example for Linux:**

```bash
huggingface-cli download <repo-id> \
  --repo-type model \
  --local-dir "$HOME/.local/share/izwi/models/<model-name>"
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

## Chatterbox

`Chatterbox-500M` (about 3.2 GB) needs five files from Hugging Face
(`ResembleAI/chatterbox` at a pinned revision) and one from outside it: the
PerTh watermarker checkpoint, `perth_net_250000.pth.tar`, from Resemble AI's
Perth repository on GitHub. Place all six files in the model's directory:

```bash
MODEL_DIR="<izwi-models-path>/Chatterbox-500M"

huggingface-cli download ResembleAI/chatterbox \
  t3_cfg.safetensors s3gen.safetensors ve.safetensors tokenizer.json conds.pt \
  --revision 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18 \
  --local-dir "$MODEL_DIR"

curl -L -o "$MODEL_DIR/perth_net_250000.pth.tar" \
  https://raw.githubusercontent.com/resemble-ai/Perth/ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar
```

When Izwi downloads the checkpoint itself, it checks the size (37,429,684
bytes) and SHA-256
(`a15bce457ebc53ce5e6c9c3f11df78cf7ee2bf9cdab0a798902135b4c4027670`). Check a
copy you place yourself with `shasum -a 256` (macOS) or `sha256sum` (Linux).
If the server runs with `IZWI_CHATTERBOX_WATERMARK=false`, the model loads
without the watermarker and the checkpoint is not needed.

`Chatterbox-Multilingual-500M` (about 3.2 GB) takes its own speech-token model
and tokenizer from the same repository and revision, with the English S3Gen,
voice encoder, default voice and the same PerTh checkpoint:

```bash
MODEL_DIR="<izwi-models-path>/Chatterbox-Multilingual-500M"

huggingface-cli download ResembleAI/chatterbox \
  t3_mtl23ls_v2.safetensors grapheme_mtl_merged_expanded_v1.json \
  s3gen.safetensors ve.safetensors conds.pt \
  --revision 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18 \
  --local-dir "$MODEL_DIR"

curl -L -o "$MODEL_DIR/perth_net_250000.pth.tar" \
  https://raw.githubusercontent.com/resemble-ai/Perth/ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar
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
