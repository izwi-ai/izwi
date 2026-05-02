Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Enter-MsvcDevEnv {
    $existingCl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if ($existingCl) {
        Write-Host "MSVC cl.exe already available: $($existingCl.Source)"
        return
    }

    $vswherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    if (-not (Test-Path $vswherePath)) {
        throw "vswhere.exe was not found; cannot locate Visual Studio Build Tools for CUDA compilation."
    }

    $installPath = & $vswherePath `
        -latest `
        -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath

    if (-not $installPath) {
        throw "Visual Studio Build Tools with the x64 C++ toolchain were not found."
    }

    $vcvarsPath = Join-Path $installPath "VC/Auxiliary/Build/vcvars64.bat"
    if (-not (Test-Path $vcvarsPath)) {
        throw "vcvars64.bat was not found at $vcvarsPath."
    }

    $envLines = & cmd.exe /s /c "`"$vcvarsPath`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "vcvars64.bat failed with exit code $LASTEXITCODE."
    }

    foreach ($line in $envLines) {
        if ($line -match '^([^=]+)=(.*)$') {
            Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
        }
    }

    $cl = Get-Command cl.exe -ErrorAction SilentlyContinue
    if (-not $cl) {
        throw "MSVC cl.exe is still not available after loading vcvars64.bat."
    }

    Write-Host "MSVC cl.exe available: $($cl.Source)"
}

Enter-MsvcDevEnv
