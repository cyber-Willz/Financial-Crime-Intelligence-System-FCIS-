# AMLSim-style archive on Windows PowerShell (v0.8):
#   1. offline back-test of the calibrated profile (no database needed)
#   2. start PostgreSQL, migrate, apply the profile, bulk-import transactions.csv through the scoring pipeline
#   3. serve the API and read back alerts / stats
# Usage:  .\scripts\run_amlsim.ps1 -Csv C:\data\transactions.csv
#         .\scripts\run_amlsim.ps1 -Csv C:\data\transactions.csv -SkipDb      # back-test only
param(
  [Parameter(Mandatory = $true)][string]$Csv,
  [string]$Profile = "$PSScriptRoot\..\profiles\amlsim_fitted.json",
  [string]$Db = "postgres://aml:aml_dev_pw@localhost:5432/aml",
  [switch]$SkipDb,
  [switch]$SkipTests
)
$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\.."
$Csv = (Resolve-Path $Csv).Path
$Profile = (Resolve-Path $Profile).Path
$env:DATABASE_URL = $Db
$env:AML_API_KEYS = "adminkey-0123456789:admin:mlro,analystkey-0123456789:analyst:bob,ingestkey-0123456789:ingest:core"
$env:AML_BIND = "127.0.0.1:8080"
$base = "http://127.0.0.1:8080"
$admin = @{ "x-api-key" = "adminkey-0123456789" }
$analyst = @{ "x-api-key" = "analystkey-0123456789" }

function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" } }

if (-not $SkipTests) { cargo test --release --workspace; Check "cargo test" }
cargo build --release; Check "cargo build"

# 1. Offline back-test: calibrated on the first 70% of days, reported on the unseen last 30%.
#    Expect precision ~99.6%, recall ~98.7% (TP=544 FP=2 on the shipped profile).
New-Item -ItemType Directory -Force -Path .\reports\out | Out-Null
.\target\release\aml-eval.exe --csv $Csv --format amlsim --config $Profile --eval-from 0.7 `
  --dump-errors .\reports\out\amlsim_errors.csv --json-out .\reports\out\amlsim_eval.json
Check "aml-eval"
Write-Host "Errors (false positives / misses) written to reports\out\amlsim_errors.csv"
if ($SkipDb) { return }

# 2. Database, schema, profile.
docker compose up -d db; Check "docker compose"
for ($i = 0; $i -lt 30; $i++) {
  docker compose exec -T db pg_isready -U aml -d aml *> $null
  if ($LASTEXITCODE -eq 0) { break }
  Start-Sleep 2
}
Check "postgres readiness"
.\target\release\aml.exe migrate --fresh; Check "migrate"

# The profile is applied through the API (audited, all-or-nothing); the server is stopped again before the bulk import
# so the importer and the server never hold diverging in-memory engine state.
$srv = Start-Process .\target\release\aml.exe -ArgumentList "serve" -PassThru -NoNewWindow
try {
  $up = $false
  for ($i = 0; $i -lt 30 -and -not $up; $i++) {
    try { Invoke-RestMethod "$base/health" | Out-Null; $up = $true } catch { Start-Sleep 1 }
  }
  if (-not $up) { throw "API did not start" }
  Invoke-RestMethod -Method Post -Uri "$base/api/v1/rules/import" -Headers $admin -ContentType "application/json" `
    -Body (Get-Content $Profile -Raw) | ConvertTo-Json -Depth 4
} finally { Stop-Process -Id $srv.Id -Force }

# 3. Bulk import (200 rows are refused by API validation: zero amounts / self-transfers), then serve.
.\target\release\aml.exe import-dataset --format amlsim --file $Csv; Check "import-dataset"

$srv = Start-Process .\target\release\aml.exe -ArgumentList "serve" -PassThru -NoNewWindow
try {
  $up = $false
  for ($i = 0; $i -lt 60 -and -not $up; $i++) {
    try { Invoke-RestMethod "$base/health" | Out-Null; $up = $true } catch { Start-Sleep 1 }
  }
  if (-not $up) { throw "API did not start" }
  Invoke-RestMethod -Uri "$base/api/v1/stats/overview" -Headers $analyst | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Uri "$base/api/v1/alerts?min_score=60&limit=5" -Headers $analyst | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Uri "$base/api/v1/rules" -Headers $analyst |
    ConvertTo-Json -Depth 6 | Select-String -Pattern "repeat_tol|AMOUNT_DROP" | Select-Object -First 4
} finally { Stop-Process -Id $srv.Id -Force }
