param(
    [switch]$AllowMissing,
    [switch]$AllowMissingDriver,
    [switch]$ExpectCudaDlls,
    [switch]$SkipStartupProbe,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Binaries
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: scripts/release/audit-runtime-deps.ps1 [-AllowMissing] [-AllowMissingDriver] [-ExpectCudaDlls] [-SkipStartupProbe] <binary>...

Audits Windows release binaries for CUDA DLL availability and verifies that
each binary can at least enter its version/help path.

Options:
  -AllowMissing       Report missing CUDA DLLs without failing the audit.
  -AllowMissingDriver Allow missing nvcuda.dll, but fail other missing CUDA DLLs.
  -ExpectCudaDlls     Require CUDA runtime DLLs other than nvcuda.dll to be visible.
  -SkipStartupProbe
                   Do not execute --version/--help after DLL inspection.
"@
}

if (-not $Binaries -or $Binaries.Count -eq 0) {
    Show-Usage
    exit 1
}

$cudaDllPatterns = @(
    "nvcuda.dll",
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "curand64_*.dll",
    "nvrtc64_*.dll",
    "cudnn64_*.dll"
)

$cudaDependencyPattern = "(?i)\b([A-Za-z0-9_.-]*(cuda|cublas|curand|nvrtc|cudnn)[A-Za-z0-9_.-]*\.dll)\b"

function Resolve-DllCandidate {
    param([string]$Pattern, [string]$BinaryDir)

    $searchDirs = @($BinaryDir)
    if ($env:PATH) {
        $searchDirs += $env:PATH.Split(";") | Where-Object { $_ -and $_.Trim() -ne "" }
    }
    if ($env:CUDA_PATH) {
        $searchDirs += Join-Path $env:CUDA_PATH "bin"
    }

    foreach ($dir in $searchDirs | Select-Object -Unique) {
        if (-not (Test-Path $dir)) {
            continue
        }
        $match = Get-ChildItem -Path $dir -Filter $Pattern -File -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($match) {
            return $match.FullName
        }
    }

    return $null
}

function Is-DriverDll {
    param([string]$Name)

    return $Name -ieq "nvcuda.dll"
}

function Get-CudaDependencyNames {
    param([object[]]$DumpbinOutput)

    $names = @()
    foreach ($line in $DumpbinOutput) {
        $text = [string]$line
        if ($text -match $cudaDependencyPattern) {
            $names += $Matches[1]
        }
    }

    return $names | Sort-Object -Unique
}

function Test-MissingAllowed {
    param([string]$Name)

    if ($AllowMissing) {
        return $true
    }

    if ($AllowMissingDriver -and (Is-DriverDll -Name $Name)) {
        return $true
    }

    return $false
}

function Invoke-StartupProbe {
    param([string]$Binary)

    Write-Host "Startup probe:"
    $env:IZWI_BACKEND = "cpu"

    $version = & $Binary --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        $version | ForEach-Object { Write-Host "  $_" }
        return $true
    }

    $help = & $Binary --help 2>&1
    if ($LASTEXITCODE -eq 0) {
        $help | Select-Object -First 8 | ForEach-Object { Write-Host "  $_" }
        return $true
    }

    Write-Host "  failed to execute --version or --help"
    return $false
}

function Find-Dumpbin {
    $direct = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($direct) {
        return $direct.Source
    }

    $vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    if (-not (Test-Path $vswherePath)) {
        return $null
    }

    $installPath = & $vswherePath `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath

    if (-not $installPath) {
        return $null
    }

    $toolsRoot = Join-Path $installPath "VC/Tools/MSVC"
    if (-not (Test-Path $toolsRoot)) {
        return $null
    }

    $candidate = Get-ChildItem -Path $toolsRoot -Recurse -Filter dumpbin.exe -File -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -match "\\bin\\Hostx64\\x64\\dumpbin\.exe$" } |
        Sort-Object FullName -Descending |
        Select-Object -First 1

    if ($candidate) {
        return $candidate.FullName
    }

    return $null
}

$failed = $false
$dumpbin = Find-Dumpbin

foreach ($binary in $Binaries) {
    Write-Host "==> $binary"

    if (-not (Test-Path $binary)) {
        Write-Error "Binary does not exist: $binary"
        $failed = $true
        continue
    }

    $resolved = (Resolve-Path $binary).Path
    $binaryDir = Split-Path -Parent $resolved

    if ($dumpbin) {
        Write-Host "dumpbin /DEPENDENTS:"
        $deps = & $dumpbin /DEPENDENTS $resolved 2>&1
        $deps | ForEach-Object { Write-Host "  $_" }

        $cudaDeps = Get-CudaDependencyNames -DumpbinOutput $deps
        foreach ($dep in $cudaDeps) {
            $found = Resolve-DllCandidate -Pattern $dep -BinaryDir $binaryDir
            if (-not $found -and -not (Test-MissingAllowed -Name $dep)) {
                Write-Host "  missing imported CUDA DLL: $dep"
                $failed = $true
            }
        }
    } else {
        Write-Host "dumpbin.exe not found; checking known CUDA DLLs on PATH and next to the binary."
    }

    Write-Host "Known CUDA DLL availability:"
    foreach ($pattern in $cudaDllPatterns) {
        $found = Resolve-DllCandidate -Pattern $pattern -BinaryDir $binaryDir
        if ($found) {
            Write-Host "  $pattern -> $found"
        } else {
            Write-Host "  $pattern -> missing"
            if ($ExpectCudaDlls -and -not (Test-MissingAllowed -Name $pattern)) {
                $failed = $true
            }
        }
    }

    if ($SkipStartupProbe) {
        Write-Host "Startup probe: skipped"
    } else {
        if (-not (Invoke-StartupProbe -Binary $resolved)) {
            $failed = $true
        }
    }

    Write-Host ""
}

if ($failed) {
    exit 1
}
