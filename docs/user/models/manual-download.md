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

`Maya1-3B-GGUF` (about 3.6 GB) takes the Q8_0 decoder from
`mradermacher/maya1-GGUF`, the config and tokenizer from `maya-research/maya1`,
and the same SNAC decoder:

```bash
MODEL_DIR="<izwi-models-path>/Maya1-3B-GGUF"

huggingface-cli download mradermacher/maya1-GGUF maya1.Q8_0.gguf \
  --revision 595e0193e6ca6d4a2e5a7ea18e3bf8ada39aae53 \
  --local-dir "$MODEL_DIR"

huggingface-cli download maya-research/maya1 config.json tokenizer.json \
  --revision b1b9b38ca9decbfcfd6c98ea93dd502bfd0f9572 \
  --local-dir "$MODEL_DIR"

curl -L -o "$MODEL_DIR/snac_24khz.pth" \
  https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec/pytorch_model.bin
```

---

## Maya 2 Native

`Maya2-Native-2.5B` (about 5.5 GB) needs seven files from Hugging Face
(`maya-research/maya2-native` at a pinned revision), the Mimi codec's config
and weights from `kyutai/mimi`, saved in `mimi/`, and SpeechBrain's ECAPA
speaker encoder from `speechbrain/spkrec-ecapa-voxceleb`, saved in `ecapa/`:

```bash
MODEL_DIR="<izwi-models-path>/Maya2-Native-2.5B"

huggingface-cli download maya-research/maya2-native \
  maya3_config.json model.safetensors.index.json \
  model-00001-of-00002.safetensors model-00002-of-00002.safetensors \
  backbone/config.json backbone/tokenizer.json voices/voices.json \
  --revision 1403b841a6af0f920781319e851d4d6bdeb596c6 \
  --local-dir "$MODEL_DIR"

curl -L --create-dirs -o "$MODEL_DIR/mimi/config.json" \
  https://huggingface.co/kyutai/mimi/resolve/89091b3e466eb6a9d11e537bf26b144f194978f7/config.json

curl -L --create-dirs -o "$MODEL_DIR/mimi/model.safetensors" \
  https://huggingface.co/kyutai/mimi/resolve/89091b3e466eb6a9d11e537bf26b144f194978f7/model.safetensors

curl -L --create-dirs -o "$MODEL_DIR/ecapa/embedding_model.ckpt" \
  https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286/embedding_model.ckpt
```

When Izwi downloads the last three files itself, it checks each one's size and
SHA-256:

| File | Size (bytes) | SHA-256 |
|------|--------------|---------|
| `mimi/config.json` | 1,117 | `aca6f44b04f7bc2e7466b71597d2d51e463ed1cf3cd7025d8848595580546c36` |
| `mimi/model.safetensors` | 384,649,828 | `bac7e85083dcded655d24eaadde7e6eea34c0da1b35fa2d284e641bd2b942a5e` |
| `ecapa/embedding_model.ckpt` | 83,316,686 | `0575cb64845e6b9a10db9bcb74d5ac32b326b8dc90352671d345e2ee3d0126a2` |

---

## Fish Audio Drama 3

`FishAudio-Drama-3` (about 41.5 GB) needs the 24 files of `fishaudio/drama-3`
at a pinned revision: 15 safetensors shards, the codec (`codec.pth`, the same
file Fish S2 Pro ships), the configs, tokenizer, chat template, `README.md`
and `LICENSE`. Place them in the model's directory:

```bash
MODEL_DIR="<izwi-models-path>/FishAudio-Drama-3"

huggingface-cli download fishaudio/drama-3 \
  --revision 78e5ab79e49aa9a25abe8799eff2c4d73a1c3216 \
  --local-dir "$MODEL_DIR"
```

When Izwi downloads the model itself, it checks every file's size and SHA-256
once, after the download. Every load checks only that each file is present at
its exact size and that `config.json` has its pinned hash, so check a copy you
place yourself with `shasum -a 256` (macOS) or `sha256sum` (Linux):

| File | Size (bytes) | SHA-256 |
|------|--------------|---------|
| `LICENSE` | 19,685 | `733a32442087acbda12ecafe8e71eb8ee5890c89b50ce7ac4ea3777faa3c823e` |
| `README.md` | 7,996 | `6d147e079b87e0b9bb4e1c899f75da3e658754933457a3ca753ca9912d1f5e10` |
| `chat_template.jinja` | 7,756 | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `codec.pth` | 1,871,099,728 | `74fc41c5a7151c6f350af8bd7e5d6e3accfcc7f3dfbfac23afd35af07052bb2f` |
| `config.json` | 5,534 | `aea53f0f2b9a15023fc0f8eb0f42360663671c99814037bada6867d508010111` |
| `model-00001-of-00015.safetensors` | 2,862,881,576 | `a1b7af15210d2eda2d75a2cf976295f80e345855b745b109aa3ce49d26a44497` |
| `model-00002-of-00015.safetensors` | 2,628,610,248 | `a95e113c6f32f8fcb23ecb618cd4384fad6358f3c0ac31cfbf352daf5370db94` |
| `model-00003-of-00015.safetensors` | 2,628,610,224 | `a34d0952d6369f5c73e6ae06138ef96bc6a3af9bb259c4e41425a33706cf5b68` |
| `model-00004-of-00015.safetensors` | 2,628,611,792 | `13ac368d87441a18752e424e485b711a115ea1084e3a18ab13276e73eedef8b4` |
| `model-00005-of-00015.safetensors` | 2,641,525,264 | `c432fa0dac97471f0d625bc67eb94fe084ca2e5ecd8865f457b751479b50e363` |
| `model-00006-of-00015.safetensors` | 2,628,614,896 | `661a31433747fa9d029632a61350ea9eb105bf7276d013ff0901b6a939fb2dcb` |
| `model-00007-of-00015.safetensors` | 2,628,614,880 | `4a7574b5c238f424b37aa945a736da1994a26bd38b86bf3ec4abf524c0969cc8` |
| `model-00008-of-00015.safetensors` | 2,628,614,888 | `7fac6d1e899939eb03c9cf28a33b0bfd9c7a599298662f26d9a59d64c96d007f` |
| `model-00009-of-00015.safetensors` | 2,641,525,264 | `fcf1313744ea5978e363b2bb49c3b66748d886d79f3ec102f4346a462385de4b` |
| `model-00010-of-00015.safetensors` | 2,628,614,896 | `6db3f7cd9d49806ae4f17b6f6d17a0c8dd9686aee57728da1b50ec0c6139a629` |
| `model-00011-of-00015.safetensors` | 2,628,614,880 | `a9027ef4db30bb1875bd2bc93b80bae785c5943807efe63bf09834b66e49fc16` |
| `model-00012-of-00015.safetensors` | 2,628,614,888 | `c16fea955e973010064ff7bc907a6fa67821956e0de23f74807c31fb48339f3a` |
| `model-00013-of-00015.safetensors` | 2,641,525,264 | `64861c8e1ee13a0cbb5ec9431f402cee33edcbcbc744ff63091fe3a997f0da19` |
| `model-00014-of-00015.safetensors` | 3,749,111,392 | `ba5319475a6258984143b69af8380f166df8179af90378f2f4e02694cad87baf` |
| `model-00015-of-00015.safetensors` | 1,409,461,016 | `1e2ddf01c1be73fb19f3022c28e90d24e730c1bc759e5de78083afb39aab3fb7` |
| `model.safetensors.index.json` | 6,880,196 | `a8e9aca4e9fddca51f03fad63f52c82c470e98e28b6b0b1b1562dbe6456b50bb` |
| `special_tokens_map.json` | 102,779 | `08aed9e321dc05275ac68cf883bfa96006b3fc09ff5cccadf262b6bce0873adb` |
| `tokenizer.json` | 20,791,070 | `0e9b9db76b2f0389b04b6ef057a2ac3b89100b7e96f2dd7034d19580a09654d6` |
| `tokenizer_config.json` | 103,580 | `00e800279d8436c324b924e97872c7eacd2b19406aa0fe71eeb557b4cd53aa3e` |

The 24 files total 41,502,569,692 bytes. The weights are licensed under the
Fish Audio Community License v1.0 (the repository's `LICENSE`); see
[Models](/models).
