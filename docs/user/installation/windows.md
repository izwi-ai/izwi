# Windows Installation

Izwi runs on Windows 10 and later with optional CUDA support for NVIDIA GPUs.

---

## Requirements

- **Windows 10** (version 1903+) or **Windows 11**
- **8 GB RAM** minimum (16 GB recommended)
- **10 GB** free disk space (more for models)
- **NVIDIA GPU** (optional, for CUDA acceleration)

---

## Install from Installer (Recommended)

### Step 1: Download

Download `Izwi-Setup-*.exe` from [GitHub Releases](https://github.com/izwi-ai/izwi/releases).

### Step 2: Run Installer

1. Double-click the downloaded `.exe` file
2. If Windows SmartScreen appears, click **More info** → **Run anyway**
3. Follow the installation wizard
4. Choose installation location (default: `C:\Program Files\Izwi`)

### Step 3: Verify

Open **Command Prompt** or **PowerShell** and run:

```powershell
izwi --version
```

---

## Optional: Install `espeak-ng` for Kokoro-82M

`Kokoro-82M` uses `espeak-ng` for phonemization. Install it before using Kokoro voices.

### Option 1: Install from eSpeak NG releases (Recommended)

1. Download the latest Windows installer/zip for **eSpeak NG** from the official releases page:
   [eSpeak NG Releases](https://github.com/espeak-ng/espeak-ng/releases)
2. Install eSpeak NG (or extract it to a folder such as `C:\Program Files\eSpeak NG\`)
3. Add the folder containing `espeak-ng.exe` to your **PATH** if the installer does not do this automatically

### Verify

Open a new PowerShell window and run:

```powershell
espeak-ng --version
```

---

## Install via winget (Coming Soon)

```powershell
winget install Agentem.Izwi
```

---

## CUDA Support (NVIDIA GPUs)

For NVIDIA GPU acceleration:

### Step 1: Install NVIDIA Drivers

Download and install the latest drivers from [NVIDIA](https://www.nvidia.com/drivers).

### Step 2: Install CUDA Toolkit

Download CUDA Toolkit from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads).

### Step 3: Verify CUDA

```powershell
izwi status --detailed
```

Look for "CUDA: available" in the output.

---

## Data Locations

| Data | Location |
|------|----------|
| **Models** | `%APPDATA%\izwi\models\` |
| **Config** | `%APPDATA%\izwi\config.toml` |
| **Logs** | `%APPDATA%\izwi\logs\` |
| **Program** | `C:\Program Files\Izwi\` |

To open these folders:

```powershell
# Open models folder
explorer "%APPDATA%\izwi\models"

# Open config folder
explorer "%APPDATA%\izwi"
```

---

## Running at Startup

### Option 1: Task Scheduler

1. Open **Task Scheduler**
2. Click **Create Basic Task**
3. Name: "Izwi Server"
4. Trigger: **When I log on**
5. Action: **Start a program**
6. Program: `C:\Program Files\Izwi\izwi.exe`
7. Arguments: `serve`

### Option 2: Startup Folder

1. Press `Win + R`, type `shell:startup`, press Enter
2. Create a shortcut to `izwi.exe serve`

---

## Uninstall

### Via Settings

1. Open **Settings → Apps → Installed apps**
2. Find **Izwi**
3. Click **Uninstall**

### Via Control Panel

1. Open **Control Panel → Programs → Uninstall a program**
2. Find **Izwi**
3. Click **Uninstall**

### Remove Data (Optional)

```powershell
# Remove all Izwi data (including models!)
Remove-Item -Recurse -Force "$env:APPDATA\izwi"
```

---

## Troubleshooting

### "Windows protected your PC" (SmartScreen)

This appears because the app isn't code-signed yet:

1. Click **More info**
2. Click **Run anyway**

### Command not found: izwi

The installer should add Izwi to your PATH. If not:

1. Open **System Properties → Environment Variables**
2. Under **User variables**, edit **Path**
3. Add `C:\Program Files\Izwi`
4. Restart your terminal

### Kokoro error: `espeak-ng not found`

Install eSpeak NG and ensure the folder containing `espeak-ng.exe` is in your PATH, then verify:

```powershell
espeak-ng --version
```

### Port 8080 already in use

Another application is using port 8080. Use a different port:

```powershell
izwi serve --port 8888
```

### CUDA not detected

1. Verify NVIDIA drivers:
   ```powershell
   nvidia-smi
   ```

2. Ensure CUDA Toolkit is installed and in PATH

3. Restart your computer after CUDA installation

### Firewall blocking connections

If you can't access Izwi from other devices:

1. Open **Windows Defender Firewall**
2. Click **Allow an app through firewall**
3. Add `izwi-server.exe`

---

## Building from Source

### Prerequisites

1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install [Rust](https://rustup.rs/)
3. Install [Git](https://git-scm.com/download/win)

### Build

```powershell
git clone https://github.com/izwi-ai/izwi.git
cd izwi
cargo build --release
```

Binaries will be in `target\release\`.

---

## Next Steps

- [Getting Started](../getting-started.md)
- [Download Models](../models/index.md)
