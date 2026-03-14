param(
  [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot ".."))
)

Set-Location $RepoRoot

git rev-parse --is-inside-work-tree > $null 2>&1
if ($LASTEXITCODE -ne 0) {
  Write-Error "Not a git repository. Run 'git init' first."
  exit 1
}

if (-not (Test-Path ".githooks")) {
  Write-Error "Missing .githooks directory."
  exit 1
}

git config core.hooksPath .githooks
Write-Host "Git hooks path set to .githooks"
