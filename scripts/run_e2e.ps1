# End-to-end run on Windows PowerShell: migrate -> import SAML-D sample -> serve -> exercise the API.
# Usage:  .\scripts\run_e2e.ps1 -Csv C:\path\SAML-D.csv -Rows 300000
param(
  [string]$Csv = "$PSScriptRoot\..\SAML-D.csv",
  [int]$Rows = 300000,
  [string]$AmlSuspicious = "",
  [string]$Db = "postgres://aml:aml_dev_pw@localhost:5432/aml"
)
$ErrorActionPreference = "Stop"
$env:DATABASE_URL = $Db
$env:AML_API_KEYS = "adminkey-0123456789:admin:mlro,analystkey-0123456789:analyst:bob,ingestkey-0123456789:ingest:core"
$env:AML_BIND = "127.0.0.1:8080"
Set-Location "$PSScriptRoot\.."

cargo test --release --workspace
cargo build --release
.\target\release\aml.exe migrate --fresh
.\target\release\aml.exe import-dataset --format saml-d --file $Csv --limit $Rows
if ($AmlSuspicious) { .\target\release\aml.exe import-dataset --format aml-suspicious --file $AmlSuspicious }

$srv = Start-Process .\target\release\aml.exe -ArgumentList "serve" -PassThru -NoNewWindow
try {
  Start-Sleep 8
  $h = @{ "x-api-key" = "ingestkey-0123456789" }
  $a = @{ "x-api-key" = "analystkey-0123456789" }
  Invoke-RestMethod http://127.0.0.1:8080/health
  $tx = @{ sender_account = 9000001; receiver_account = 9000003; amount = 2500000; payment_currency = "GBP";
           received_currency = "NGN"; sender_country = "GB"; receiver_country = "NG"; payment_type = "CROSS_BORDER" } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/api/v1/transactions -Headers $h -ContentType "application/json" -Body $tx | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/alerts?min_score=80&limit=5" -Headers $a | ConvertTo-Json -Depth 5
  Invoke-RestMethod -Uri http://127.0.0.1:8080/api/v1/stats/overview -Headers $a | ConvertTo-Json -Depth 5
} finally { Stop-Process -Id $srv.Id -Force }

# --- v0.6: agent + feedback loop smoke test (deterministic analyst; set AML_LLM_CMD/AML_LLM_MODEL to attach a model)
try {
  $first = (Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/alerts?limit=1" -Headers $a).items[0]
  if ($first) {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/api/v1/agent/alerts/$($first.id)/explain" -Headers $a | ConvertTo-Json -Depth 6
    Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/alerts/queue" -Headers $a | ConvertTo-Json -Depth 4
    Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/feedback/summary" -Headers $a | ConvertTo-Json -Depth 4
  }
} catch { Write-Warning "v0.6 smoke test: $_" }
