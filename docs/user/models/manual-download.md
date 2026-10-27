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

`Chatterbox-Multilingual-500M` (about 3.3 GB) takes its own speech-token model
and tokenizer from the same repository and revision, with the English S3Gen,
voice encoder, default voice and the same PerTh checkpoint. Chinese also needs
the repository's Cangjie table and two files of spacy-pkuseg's word segmenter
from GitHub: its default model, `spacy_ontonotes.zip` (34,567,143 bytes,
SHA-256 `b216e7f92de7ae285aeab8feba2faa8ea8216e5995ff6fb3d391cc8356db1bfe`),
and its merge dictionary, saved as `pkuseg_default.pkl` (4,525,959 bytes,
SHA-256 `3cf450d95719d51883fa49291302b9032372064d86805b130bb2ea0223196c67`):

```bash
MODEL_DIR="<izwi-models-path>/Chatterbox-Multilingual-500M"

huggingface-cli download ResembleAI/chatterbox \
  t3_mtl23ls_v2.safetensors grapheme_mtl_merged_expanded_v1.json \
  s3gen.safetensors ve.safetensors conds.pt Cangjie5_TC.json \
  --revision 5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18 \
  --local-dir "$MODEL_DIR"

curl -L -o "$MODEL_DIR/perth_net_250000.pth.tar" \
  https://raw.githubusercontent.com/resemble-ai/Perth/ff1c8ac55a976971245cdd53c18d6131ca00d993/src/perth/perth_net/pretrained/implicit/perth_net_250000.pth.tar

curl -L -o "$MODEL_DIR/spacy_ontonotes.zip" \
  https://github.com/explosion/spacy-pkuseg/releases/download/v0.0.26/spacy_ontonotes.zip

curl -L -o "$MODEL_DIR/pkuseg_default.pkl" \
  https://raw.githubusercontent.com/explosion/spacy-pkuseg/16995617f3b8f5cf32e5b6a77d7d800e201b255b/spacy_pkuseg/dicts/default.pkl
```

Izwi reads the dictionary as the one pickled string it holds and never
unpickles it.

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

---

## Maya1

`Maya1-3B` (about 6.7 GB) needs five files from Hugging Face
(`maya-research/maya1` at a pinned revision) and the SNAC 24 kHz decoder from
`hubertsiuzdak/snac_24khz`, saved as `snac_24khz.pth` beside them:

```bash
MODEL_DIR="<izwi-models-path>/Maya1-3B"

huggingface-cli download maya-research/maya1 \
  config.json model.safetensors.index.json tokenizer.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  --revision b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572 \
  --local-dir "$MODEL_DIR"

curl -L -o "$MODEL_DIR/snac_24khz.pth" \
  https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec/pytorch_model.bin
```

When Izwi downloads the decoder itself, it checks the size (79,488,254 bytes)
and SHA-256
(`4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40`).
