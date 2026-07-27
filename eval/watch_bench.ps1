<#
.SYNOPSIS
  Live viewer for eval/benchmarks.py -- streams per-question answers and running
  accuracy from eval/reports/benchmarks_live.jsonl (written on C: via /mnt/c).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File eval\watch_bench.ps1
  powershell -ExecutionPolicy Bypass -File eval\watch_bench.ps1 -Once
#>
param(
  [string]$Repo  = "",
  [string]$Live  = "",
  [int]$Interval = 5,
  [switch]$Once
)

# Repo root from this script's own location, so the path is not pinned to one
# machine. Override with -Repo or $env:BECUSSY_REPO.
if (-not $Repo) {
  if ($env:BECUSSY_REPO) { $Repo = $env:BECUSSY_REPO }
  else { $Repo = Split-Path -Parent $PSScriptRoot }
}
if (-not $Live) { $Live = Join-Path $Repo 'eval\reports\benchmarks_live.jsonl' }

function Show-Bench {
  Clear-Host
  Write-Host "=== Becussy benchmark monitor ===  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
  if (-not (Test-Path $Live)) {
    Write-Host "no live log yet -- start a run: eval/benchmarks.py (writes $Live)" -ForegroundColor DarkGray
    return
  }
  $rows = @(Get-Content $Live -ErrorAction SilentlyContinue | ForEach-Object {
    try { $_ | ConvertFrom-Json } catch {} })
  if ($rows.Count -eq 0) { Write-Host "(waiting for first question...)" -ForegroundColor DarkGray; return }

  Write-Host "`nrunning accuracy:" -ForegroundColor Cyan
  $rows | Group-Object tag, task | ForEach-Object {
    $c = ($_.Group | Where-Object { $_.correct }).Count
    $t = $_.Group.Count
    $acc = if ($t) { [math]::Round(100.0 * $c / $t, 1) } else { 0 }
    "{0,-22} {1,4}/{2,-4}  acc {3,5}%" -f $_.Name, $c, $t, $acc | Write-Host
  }

  Write-Host "`nlast 8 answers:" -ForegroundColor Cyan
  $rows | Select-Object -Last 8 | ForEach-Object {
    $mark = if ($_.correct) { "[OK] " } else { "[x]  " }
    $color = if ($_.correct) { "Green" } else { "DarkGray" }
    $ans = ($_.output -replace '\s+', ' ')
    if ($ans.Length -gt 150) { $ans = $ans.Substring(0, 150) + "..." }
    Write-Host ("{0}{1}/{2} {3}  got={4} gold={5}" -f $mark, $_.tag, $_.task, ("#" + $_.i), $_.extracted, $_.gold) -ForegroundColor $color
    Write-Host ("      {0}" -f $ans) -ForegroundColor DarkGray
  }
}

if ($Once) { Show-Bench; return }
Write-Host "watching benchmark (Ctrl-C to stop)..." -ForegroundColor DarkGray
while ($true) { Show-Bench; Start-Sleep -Seconds $Interval }
