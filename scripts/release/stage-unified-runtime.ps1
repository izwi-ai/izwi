param(
    [string]$ReleaseDir = "target/release",
    [string]$CudaTargetDir = "target/cuda-release",
    [switch]$CopyCudaLibs,
    [string]$CudaBinDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$cudaReleaseDir = Join-Path $CudaTargetDir "release"
$runtimeDir = Join-Path $ReleaseDir "runtime/cuda"
$requiredBins = @("izwi.exe", "izwi-server.exe")

foreach ($bin in $requiredBins) {
    $publicPath = Join-Path $ReleaseDir $bin
    $cudaPath = Join-Path $cudaReleaseDir $bin

    if (-not (Test-Path $publicPath)) {
        throw "Missing public release binary: $publicPath"
    }
    if (-not (Test-Path $cudaPath)) {
        throw "Missing CUDA runtime binary: $cudaPath"
    }
}

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

foreach ($bin in $requiredBins) {
    Copy-Item -Force -Path (Join-Path $cudaReleaseDir $bin) -Destination (Join-Path $runtimeDir $bin)
}

function Resolve-CudaBinDir {
    $candidates = @()
    if ($CudaBinDir) {
        $candidates += $CudaBinDir
    }
    if ($env:CUDA_PATH) {
        $candidates += (Join-Path $env:CUDA_PATH "bin")
    }
    if ($env:CUDA_HOME) {
        $candidates += (Join-Path $env:CUDA_HOME "bin")
    }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    return $null
}

if ($CopyCudaLibs) {
    $sourceBinDir = Resolve-CudaBinDir
    if (-not $sourceBinDir) {
        throw "Could not find CUDA bin directory. Pass -CudaBinDir or set CUDA_PATH."
    }

    $patterns = @(
        "cudart64_*.dll",
        "cublas64_*.dll",
        "cublasLt64_*.dll",
        "curand64_*.dll",
        "nvrtc64_*.dll",
        "nvrtc-builtins64_*.dll"
    )

    $copied = $false
    foreach ($pattern in $patterns) {
        $matches = Get-ChildItem -Path $sourceBinDir -Filter $pattern -File -ErrorAction SilentlyContinue
        foreach ($match in $matches) {
            Copy-Item -Force -Path $match.FullName -Destination (Join-Path $runtimeDir $match.Name)
            $copied = $true
        }
    }

    if (-not $copied) {
        throw "No CUDA runtime DLLs matched in $sourceBinDir"
    }
}

Write-Host "Staged unified runtime assets in $runtimeDir"
