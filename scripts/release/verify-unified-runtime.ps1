param(
    [string]$ReleaseDir = "target/release",
    [switch]$SkipConfigCheck,
    [switch]$SkipCudaCompiledCheck,
    [switch]$SkipCudaLibraryCheck
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "../..")
$releasePath = Resolve-Path $ReleaseDir
$runtimeDir = Join-Path $releasePath "runtime/cuda"
$auditScript = Join-Path $scriptDir "audit-runtime-deps.ps1"
$publicBins = @("izwi.exe", "izwi-server.exe")
$forbiddenBins = @("izwi-cuda.exe", "izwi-server-cuda.exe")
$cudaDependencyPattern = "(?i)(nvcuda|cudart|cublas|cublasLt|curand|nvrtc|cudnn).*\.dll"

function Assert-File {
    param([string]$Path)

    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing file: $Path"
    }
}

function Assert-NoForbiddenPublicNames {
    foreach ($bin in $forbiddenBins) {
        $publicPath = Join-Path $releasePath $bin
        $privatePath = Join-Path $runtimeDir $bin
        if (Test-Path $publicPath) {
            throw "Public CUDA-suffixed binary is not allowed: $publicPath"
        }
        if (Test-Path $privatePath) {
            throw "Private runtime must keep the public basename, found: $privatePath"
        }
    }
}

function Get-CompiledBackendLines {
    param([string]$Binary)

    $output = & $Binary version --full 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to run '$Binary version --full': $output"
    }

    $lines = @()
    $inSection = $false
    foreach ($line in $output) {
        if ($line -match "^Compiled Backends:") {
            $inSection = $true
            continue
        }
        if ($inSection -and [string]::IsNullOrWhiteSpace($line)) {
            break
        }
        if ($inSection) {
            $lines += [string]$line
        }
    }

    return $lines
}

function Test-BackendSectionHasCuda {
    param([string[]]$Lines)

    foreach ($line in $Lines) {
        if ($line -match "\bCUDA\b") {
            return $true
        }
    }

    return $false
}

function Assert-CliBackendContract {
    $publicCli = Join-Path $releasePath "izwi.exe"
    $privateCli = Join-Path $runtimeDir "izwi.exe"

    $publicBackends = Get-CompiledBackendLines -Binary $publicCli
    Write-Host "Public CLI compiled backends:"
    $publicBackends | ForEach-Object { Write-Host "  $_" }
    if (Test-BackendSectionHasCuda -Lines $publicBackends) {
        throw "Public izwi.exe is CUDA-compiled; public entrypoints must remain CPU-safe."
    }

    if ($SkipCudaCompiledCheck) {
        Write-Host "Skipping private CLI CUDA compiled-backend check."
        return
    }

    $oldPath = $env:PATH
    try {
        $env:PATH = "$runtimeDir;$oldPath"
        $privateBackends = Get-CompiledBackendLines -Binary $privateCli
    } finally {
        $env:PATH = $oldPath
    }
    Write-Host "Private CUDA CLI compiled backends:"
    $privateBackends | ForEach-Object { Write-Host "  $_" }
    if (-not (Test-BackendSectionHasCuda -Lines $privateBackends)) {
        throw "Private runtime izwi.exe does not report CUDA in compiled backends."
    }
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

function Assert-NoPublicCudaDependencies {
    param([string]$Binary)

    $dumpbin = Find-Dumpbin
    if (-not $dumpbin) {
        throw "dumpbin.exe was not found; cannot verify public Windows CUDA DLL dependencies."
    }

    $deps = & $dumpbin /DEPENDENTS $Binary 2>&1
    $cudaDeps = $deps | Where-Object { $_ -match $cudaDependencyPattern }
    if ($cudaDeps) {
        $cudaDeps | ForEach-Object { Write-Host "  $_" }
        throw "Public binary has loader-visible CUDA DLL dependencies: $Binary"
    }
}

function Assert-RequiredCudaLibraries {
    if ($SkipCudaLibraryCheck) {
        Write-Host "Skipping packaged CUDA DLL check."
        return
    }

    $patterns = @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "curand64_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll"
    )

    foreach ($pattern in $patterns) {
        $matches = Get-ChildItem -Path $runtimeDir -Filter $pattern -File -ErrorAction SilentlyContinue
        if (-not $matches) {
            throw "Missing packaged CUDA runtime DLL matching $pattern in $runtimeDir"
        }
    }
}

function Assert-WindowsTauriConfig {
    if ($SkipConfigCheck) {
        Write-Host "Skipping Tauri release config check."
        return
    }

    $configPath = Join-Path $repoRoot "crates/izwi-desktop/tauri.release.windows.conf.json"
    if (-not (Test-Path $configPath)) {
        throw "Missing Windows Tauri release config: $configPath"
    }

    $config = Get-Content -Raw $configPath | ConvertFrom-Json
    $resources = $config.bundle.resources.PSObject.Properties

    function Get-ResourceValue {
        param([string]$Name)
        $property = $resources[$Name]
        if (-not $property) {
            return $null
        }
        return $property.Value
    }

    if ((Get-ResourceValue "../../target/release/izwi.exe") -ne "bin/izwi.exe") {
        throw "Windows Tauri config must bundle ../../target/release/izwi.exe as bin/izwi.exe"
    }
    if ((Get-ResourceValue "../../target/release/izwi-server.exe") -ne "bin/izwi-server.exe") {
        throw "Windows Tauri config must bundle ../../target/release/izwi-server.exe as bin/izwi-server.exe"
    }
    if ((Get-ResourceValue "../../target/release/runtime") -ne "bin/runtime") {
        throw "Windows Tauri config must bundle ../../target/release/runtime as bin/runtime"
    }
}

if (-not (Test-Path $runtimeDir -PathType Container)) {
    throw "Private CUDA runtime directory does not exist: $runtimeDir"
}
Assert-File -Path $auditScript

foreach ($bin in $publicBins) {
    Assert-File -Path (Join-Path $releasePath $bin)
    Assert-File -Path (Join-Path $runtimeDir $bin)
}

Assert-NoForbiddenPublicNames
Assert-CliBackendContract

foreach ($bin in $publicBins) {
    Assert-NoPublicCudaDependencies -Binary (Join-Path $releasePath $bin)
}

Assert-RequiredCudaLibraries
Assert-WindowsTauriConfig

Write-Host "Auditing public CPU-safe binaries:"
& $auditScript (Join-Path $releasePath "izwi.exe") (Join-Path $releasePath "izwi-server.exe")
if ($LASTEXITCODE -ne 0) {
    throw "Public runtime dependency audit failed."
}

Write-Host "Auditing private CUDA runtime binaries:"
$oldPath = $env:PATH
try {
    $env:PATH = "$runtimeDir;$oldPath"
    & $auditScript -AllowMissingDriver -ExpectCudaDlls -SkipStartupProbe `
        (Join-Path $runtimeDir "izwi.exe") `
        (Join-Path $runtimeDir "izwi-server.exe")
    if ($LASTEXITCODE -ne 0) {
        throw "Private CUDA runtime dependency audit failed."
    }
} finally {
    $env:PATH = $oldPath
}

Write-Host "Unified runtime verification passed for $releasePath"
