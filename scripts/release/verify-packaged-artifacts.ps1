param(
    [string]$TerminalZip,
    [string]$InstallerPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: scripts/release/verify-packaged-artifacts.ps1 [-TerminalZip <path>] [-InstallerPath <path>]

Verifies final Windows release artifacts contain the unified CPU+CUDA runtime
layout after packaging, not only in target/release.
"@
}

if (-not $TerminalZip -and -not $InstallerPath) {
    Show-Usage
    exit 1
}

$requiredRuntimePatterns = @(
    "runtime/cuda/izwi\.exe$",
    "runtime/cuda/izwi-server\.exe$",
    "runtime/cuda/cudart64_.*\.dll$",
    "runtime/cuda/cublas64_.*\.dll$",
    "runtime/cuda/curand64_.*\.dll$",
    "runtime/cuda/nvrtc64_.*\.dll$"
)

function Assert-File {
    param([string]$Path)

    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing artifact: $Path"
    }
}

function Normalize-Entry {
    param([string]$Entry)

    return $Entry.Replace("\", "/").TrimStart("/")
}

function Assert-ContainsPattern {
    param(
        [string[]]$Entries,
        [string]$Pattern,
        [string]$Label
    )

    $match = $Entries | Where-Object { $_ -match $Pattern } | Select-Object -First 1
    if (-not $match) {
        Write-Host "Artifact contents:"
        $Entries | ForEach-Object { Write-Host "  $_" }
        throw "Missing $Label"
    }
}

function Assert-UnifiedRuntimeEntries {
    param([string[]]$Entries)

    foreach ($pattern in $requiredRuntimePatterns) {
        Assert-ContainsPattern -Entries $Entries -Pattern $pattern -Label $pattern
    }
}

function Verify-TerminalZip {
    param([string]$Path)

    Assert-File -Path $Path
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $Path).Path)
    try {
        $entries = $zip.Entries | ForEach-Object { Normalize-Entry -Entry $_.FullName }
    } finally {
        $zip.Dispose()
    }

    Assert-ContainsPattern -Entries $entries -Pattern '(^|/)izwi\.exe$' -Label "public CLI"
    Assert-ContainsPattern -Entries $entries -Pattern '(^|/)izwi-server\.exe$' -Label "public server"
    Assert-ContainsPattern -Entries $entries -Pattern '(^|/)izwi-desktop\.exe$' -Label "desktop binary"
    Assert-UnifiedRuntimeEntries -Entries $entries

    Write-Host "Verified terminal zip: $Path"
}

function Find-7Zip {
    $direct = Get-Command 7z.exe -ErrorAction SilentlyContinue
    if ($direct) {
        return $direct.Source
    }

    $programFiles = @($env:ProgramFiles, ${env:ProgramFiles(x86)}) | Where-Object { $_ }
    foreach ($root in $programFiles) {
        $candidate = Join-Path $root "7-Zip/7z.exe"
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

function Expand-With7Zip {
    param(
        [string]$SevenZip,
        [string]$Archive,
        [string]$Destination
    )

    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & $SevenZip x -y "-o$Destination" $Archive | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "7-Zip failed to extract $Archive"
    }
}

function Verify-Installer {
    param([string]$Path)

    Assert-File -Path $Path
    $sevenZip = Find-7Zip
    if (-not $sevenZip) {
        throw "7z.exe is required to inspect Windows installer payloads."
    }

    $root = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString("N"))
    try {
        Expand-With7Zip -SevenZip $sevenZip -Archive (Resolve-Path $Path).Path -Destination $root

        $nestedArchives = Get-ChildItem -Path $root -Recurse -File -Filter "*.7z" -ErrorAction SilentlyContinue
        foreach ($archive in $nestedArchives) {
            $destination = Join-Path $archive.DirectoryName "$($archive.BaseName)-expanded"
            Expand-With7Zip -SevenZip $sevenZip -Archive $archive.FullName -Destination $destination
        }

        $entries = Get-ChildItem -Path $root -Recurse -File |
            ForEach-Object { Normalize-Entry -Entry $_.FullName.Substring($root.Length) }

        Assert-ContainsPattern -Entries $entries -Pattern '(^|/)bin/izwi\.exe$' -Label "installer public CLI resource"
        Assert-ContainsPattern -Entries $entries -Pattern '(^|/)bin/izwi-server\.exe$' -Label "installer public server resource"
        Assert-UnifiedRuntimeEntries -Entries $entries

        Write-Host "Verified Windows installer payload: $Path"
    } finally {
        if (Test-Path $root) {
            Remove-Item -Recurse -Force $root
        }
    }
}

if ($TerminalZip) {
    Verify-TerminalZip -Path $TerminalZip
}

if ($InstallerPath) {
    Verify-Installer -Path $InstallerPath
}
