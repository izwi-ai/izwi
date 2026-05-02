param(
    [string]$ArtifactDir,
    [string]$TerminalZip,
    [string]$InstallerPath,
    [Int64]$MaxBytes = 1073741824
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Usage {
    @"
Usage: scripts/release/verify-native-artifacts.ps1 [-ArtifactDir <dir>] [-TerminalZip <path>] [-InstallerPath <path>] [-MaxBytes <bytes>]

Verifies native Windows release artifacts are CPU-only and stay within a safe
size limit before upload/publish.
"@
}

if (-not $ArtifactDir -and -not $TerminalZip -and -not $InstallerPath) {
    Show-Usage
    exit 1
}

$forbiddenPattern = "runtime/cuda|libcuda\.so|libcudart\.so|libcublas(Lt)?\.so|libcurand\.so|libnvrtc(-builtins)?\.so|nvcuda\.dll|cudart64_|cublas(Lt)?64_|curand64_|nvrtc(-builtins)?64_"

function Assert-File {
    param([string]$Path)

    if (-not (Test-Path $Path -PathType Leaf)) {
        throw "Missing artifact: $Path"
    }
}

function Assert-SizeOk {
    param([string]$Path)

    $item = Get-Item -LiteralPath $Path
    if ($item.Length -gt $MaxBytes) {
        throw "$Path is $($item.Length) bytes; max allowed is $MaxBytes"
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

function Assert-NoForbiddenEntries {
    param(
        [string[]]$Entries,
        [string]$Label
    )

    $matches = $Entries | Where-Object { $_ -match $forbiddenPattern }
    if ($matches) {
        Write-Host "Forbidden CUDA/native-runtime entries in ${Label}:"
        $matches | ForEach-Object { Write-Host "  $_" }
        throw "$Label contains CUDA runtime payload"
    }
}

function Verify-TerminalZip {
    param([string]$Path)

    Assert-File -Path $Path
    Assert-SizeOk -Path $Path
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path $Path).Path)
    try {
        $entries = $zip.Entries | ForEach-Object { Normalize-Entry -Entry $_.FullName }
    } finally {
        $zip.Dispose()
    }

    Assert-NoForbiddenEntries -Entries $entries -Label $Path
    Assert-ContainsPattern -Entries $entries -Pattern '(^|/)izwi\.exe$' -Label "public CLI"
    Assert-ContainsPattern -Entries $entries -Pattern '(^|/)izwi-server\.exe$' -Label "public server"
    Assert-ContainsPattern -Entries $entries -Pattern '(^|/)izwi-desktop\.exe$' -Label "desktop binary"

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
    Assert-SizeOk -Path $Path
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

        Assert-NoForbiddenEntries -Entries $entries -Label $Path
        Assert-ContainsPattern -Entries $entries -Pattern '(^|/)bin/izwi\.exe$' -Label "installer public CLI resource"
        Assert-ContainsPattern -Entries $entries -Pattern '(^|/)bin/izwi-server\.exe$' -Label "installer public server resource"

        Write-Host "Verified Windows installer payload: $Path"
    } finally {
        if (Test-Path $root) {
            Remove-Item -Recurse -Force $root
        }
    }
}

function Verify-ArtifactFile {
    param([string]$Path)

    Assert-File -Path $Path
    Assert-SizeOk -Path $Path
    $name = Split-Path -Leaf $Path

    if ($name -like "*.zip") {
        Verify-TerminalZip -Path $Path
    } elseif ($name -match "-setup\.exe$") {
        Verify-Installer -Path $Path
    } else {
        Write-Host "Verified size-only artifact: $Path"
    }
}

if ($ArtifactDir) {
    if (-not (Test-Path $ArtifactDir -PathType Container)) {
        throw "Missing artifact directory: $ArtifactDir"
    }

    Get-ChildItem -Path $ArtifactDir -File | ForEach-Object {
        Verify-ArtifactFile -Path $_.FullName
    }
}

if ($TerminalZip) {
    Verify-TerminalZip -Path $TerminalZip
}

if ($InstallerPath) {
    Verify-Installer -Path $InstallerPath
}
