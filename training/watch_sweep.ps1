<#
.SYNOPSIS
  Live dashboard for the Becussy v2 hyperparameter sweep, native to PowerShell.

.DESCRIPTION
  Reads outputs/sweep.log and eval/reports/sweep_summary.csv (both on C:, since
  the sweep writes them via /mnt/c) plus a quick nvidia-smi each tick. Shows the
  current config, a leaderboard of scored checkpoints so far, and GPU status.
  Refreshes until the sweep reports complete. Ctrl-C to stop watching (the sweep
  keeps running in the background).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File training\watch_sweep.ps1
  powershell -ExecutionPolicy Bypass -File training\watch_sweep.ps1 -Interval 10 -Once
#>
param(
  [string]$Log      = "C:\Users\Rivian\Documents\GitHub\Becussy\outputs\sweep.log",
  [string]$Csv      = "C:\Users\Rivian\Documents\GitHub\Becussy\eval\reports\sweep_summary.csv",
  [int]$Interval    = 15,
  [int]$TotalConfigs = 12,
  [switch]$Once
)

function Show-Dashboard {
  Clear-Host
  Write-Host "=== Becussy v2 sweep monitor ===  $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan

  $complete = $false
  if (Test-Path $Log) {
    $lines = Get-Content $Log -ErrorAction SilentlyContinue
    $cur = $lines | Select-String -Pattern '=====\s*\[(\d+)/(\d+)\]\s+(\S+)' | Select-Object -Last 1
    if ($cur) {
      $m = $cur.Matches[0].Groups
      Write-Host ("current: config {0}/{1}  {2}" -f $m[1].Value, $m[2].Value, $m[3].Value) -ForegroundColor Yellow
    }
    $complete = [bool]($lines | Select-String -Pattern 'SWEEP COMPLETE' -Quiet)
    $winner = $lines | Select-String -Pattern 'WINNER:' | Select-Object -Last 1
    if ($winner) { Write-Host $winner.Line.Trim() -ForegroundColor Green }
    # last few eval-result lines, so you see activity between config boundaries
    $recent = $lines | Select-String -Pattern 'step \d+: \{' | Select-Object -Last 3
    if ($recent) {
      Write-Host "`nrecent evals:" -ForegroundColor DarkGray
      $recent | ForEach-Object { Write-Host ("  " + $_.Line.Trim()) -ForegroundColor DarkGray }
    }
  } else {
    Write-Host "waiting for $Log ..." -ForegroundColor DarkGray
  }

  if (Test-Path $Csv) {
    $rows = @(Import-Csv $Csv)
    $doneConfigs = ($rows | Select-Object -ExpandProperty config -Unique).Count
    Write-Host ("`nresults: {0} checkpoints scored across {1}/{2} configs" -f $rows.Count, $doneConfigs, $TotalConfigs)
    Write-Host "leaderboard (top 10 by score):" -ForegroundColor Cyan
    $rows |
      Sort-Object { [double]$_.score } -Descending |
      Select-Object -First 10 config, step, pivot_rate, inversion_rate, engagement, competence, distinct2, gates_ok, score |
      Format-Table -AutoSize | Out-String | Write-Host
  } else {
    Write-Host "`n(no scored checkpoints yet - first config still training)" -ForegroundColor DarkGray
  }

  try {
    $g = (wsl -d Ubuntu-24.04 -- nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>$null)
    $p = $g -split ',' | ForEach-Object { $_.Trim() }
    Write-Host ("GPU: {0}% util | {1}/{2} MiB | {3}C" -f $p[0], $p[1], $p[2], $p[3]) -ForegroundColor DarkCyan
  } catch { Write-Host "GPU: unavailable" -ForegroundColor DarkCyan }

  return $complete
}

if ($Once) { [void](Show-Dashboard); return }

Write-Host "watching sweep (Ctrl-C to stop; sweep keeps running)..." -ForegroundColor DarkGray
while ($true) {
  $done = Show-Dashboard
  if ($done) { Write-Host "`n*** SWEEP COMPLETE - see leaderboard above and v2_best ***" -ForegroundColor Green; break }
  Start-Sleep -Seconds $Interval
}
