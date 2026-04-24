param(
    [switch]$AllowMissing,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Binaries
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: scripts/release/audit-runtime-deps.ps1 [-AllowMissing] <binary>...

Audits Windows release binaries for CUDA DLL availability and verifies that
each binary can at least enter its version/help path.
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

$failed = $false
$dumpbin = Get-Command dumpbin.exe -ErrorAction SilentlyContinue

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
        $deps = & $dumpbin.Source /DEPENDENTS $resolved 2>&1
        $deps | ForEach-Object { Write-Host "  $_" }
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
            if (-not $AllowMissing -and $pattern -ne "nvcuda.dll") {
                $failed = $true
            }
        }
    }

    if (-not (Invoke-StartupProbe -Binary $resolved)) {
        $failed = $true
    }

    Write-Host ""
}

if ($failed) {
    exit 1
}
