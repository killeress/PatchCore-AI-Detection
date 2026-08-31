[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidatePattern('^\d{4}\.\d{2}\.\d{2}\.\d+$')]
    [string]$Version,

    [string]$OutputDirectory = "deployment"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

Push-Location $repoRoot
try {
    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if (-not $gitCommand) {
        throw "Git was not found. Install Git and reopen PowerShell."
    }

    $pythonPath = $null
    $pythonPrefix = @()
    $pythonCandidates = @(
        @{ Name = "python"; Prefix = @() },
        @{ Name = "python3"; Prefix = @() },
        @{ Name = "py"; Prefix = @("-3") }
    )
    foreach ($candidate in $pythonCandidates) {
        $candidateCommand = Get-Command $candidate.Name -ErrorAction SilentlyContinue
        if (-not $candidateCommand) {
            continue
        }
        $candidatePrefix = @($candidate.Prefix)
        & $candidateCommand.Source @candidatePrefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonPath = $candidateCommand.Source
            $pythonPrefix = $candidatePrefix
            break
        }
    }
    if (-not $pythonPath) {
        throw "Python 3.10 or newer was not found. Install it and reopen PowerShell."
    }

    $insideWorktree = & $gitCommand.Source rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -ne 0 -or $insideWorktree.Trim() -ne "true") {
        throw "The project directory is not a valid Git working tree: $repoRoot"
    }

    $worktreeStatus = @(& $gitCommand.Source status --porcelain --untracked-files=all)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not read the Git working tree status."
    }
    if ($worktreeStatus.Count -gt 0) {
        $details = $worktreeStatus -join [Environment]::NewLine
        throw "Git working tree is not clean. Commit or stash changes before packaging:`n$details"
    }

    $outputRoot = if ([IO.Path]::IsPathRooted($OutputDirectory)) {
        [IO.Path]::GetFullPath($OutputDirectory)
    } else {
        [IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDirectory))
    }
    $artifact = Join-Path $outputRoot "patchcore_ai_release_${Version}_codeonly.zip"
    if (Test-Path -LiteralPath $artifact) {
        throw "The release ZIP already exists. Use a new version or move the old file first: $artifact"
    }

    $builderArguments = @(
        "scripts/build_deploy_zip.py",
        "--no-backbone",
        "--version", $Version,
        "--output-dir", $OutputDirectory
    )
    & $pythonPath @pythonPrefix @builderArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Code-only packaging failed with exit code $LASTEXITCODE."
    }
    if (-not (Test-Path -LiteralPath $artifact -PathType Leaf)) {
        throw "The packager finished without creating the expected ZIP: $artifact"
    }

    Write-Host ""
    Write-Host "Code-only ZIP created: $artifact" -ForegroundColor Green
} finally {
    Pop-Location
}
