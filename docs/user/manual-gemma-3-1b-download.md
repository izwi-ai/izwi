---
title: "Manual Download: Gemma 3 1B Instruct"
description: "Download and install the gated Gemma 3 1B Instruct model manually for Izwi."
sidebarTitle: "Gemma 3 1B Manual Download"
icon: "key-round"
---
# Manual Download: Gemma 3 1B Instruct

Gemma 3 1B Instruct (`Gemma-3-1b-it`) is a powerful compact language model from Google that requires manual download due to licensing requirements.

---

## Why Manual Download?

Google's Gemma models require you to:

1. Have a Hugging Face account
2. Accept the Gemma license agreement
3. Use authenticated downloads

This cannot be done automatically by Izwi.

---

## Prerequisites

### Create a Hugging Face Account

If you don't have one, sign up at [huggingface.co](https://huggingface.co/join).

### Accept the Gemma License

1. Visit the [Gemma 3 1B model page](https://huggingface.co/google/gemma-3-1b-it)
2. Click **Agree and access repository**
3. Read and accept the license terms

> **Note:** License approval is usually instant but may take a few minutes.

---

## Step 1: Install Hugging Face CLI

Choose one method:

**Option 1: Using pipx (Recommended)**

```bash
pipx install huggingface_hub
```

**Option 2: Using pip**

```bash
python3 -m pip install --upgrade "huggingface_hub[cli]"
```

---

## Step 2: Authenticate

Log in to Hugging Face from your terminal:

```bash
huggingface-cli login
```

When prompted:
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** permissions
3. Paste the token into your terminal

Verify you're logged in:

```bash
huggingface-cli whoami
```

---

## Step 3: Download the Model

### macOS

```bash
huggingface-cli download google/gemma-3-1b-it \
  --repo-type model \
  --local-dir "$HOME/Library/Application Support/izwi/models/Gemma-3-1b-it"
```

### Linux

```bash
huggingface-cli download google/gemma-3-1b-it \
  --repo-type model \
  --local-dir "$HOME/.local/share/izwi/models/Gemma-3-1b-it"
```

### Windows (PowerShell)

```powershell
huggingface-cli download google/gemma-3-1b-it `
  --repo-type model `
  --local-dir "$env:APPDATA\izwi\models\Gemma-3-1b-it"
```

The download is approximately **2.5 GB** and may take several minutes depending on your connection.

---

## Step 4: Verify Installation

Restart Izwi or refresh the model list:

```bash
izwi list --local
```

You should see `Gemma-3-1b-it` in the list.

Load the model:

```bash
izwi models load Gemma-3-1b-it
```

---

## Troubleshooting

### "Access to model is restricted"

You haven't accepted the license yet:

1. Visit [huggingface.co/google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
2. Click **Agree and access repository**
3. Wait a few minutes for approval
4. Try the download again

### "401 Client Error: Unauthorized"

Your token doesn't have the right permissions:

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** access
3. Run `huggingface-cli login` again with the new token

### Model not appearing in Izwi

1. Check the model was downloaded to the correct directory
2. Verify the folder name is exactly `Gemma-3-1b-it`
3. Restart the Izwi server

### Download interrupted

The Hugging Face CLI automatically resumes. Just run the same download command again.

---

## Using Gemma 3 1B

Once loaded, you can use Gemma for chat:

```bash
izwi chat --model Gemma-3-1b-it
```

Or via the web UI at `http://localhost:8080/chat`.

---

## See Also

- [Models Overview](./models/index.md)
- [Manual Download Guide](./models/manual-download.md)
- [Troubleshooting](./troubleshooting.md)
