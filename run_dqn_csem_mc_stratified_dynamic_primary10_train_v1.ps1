param(
  [string]$ProjectRoot = ".",
  [string]$Python = "python",
  [string]$Primary10Manifest = "outputs\tasksets\mc_stratified_dynamic_v1\mc_stratified_dynamic_primary10_v1.csv",
  [string]$OutputRoot = "outputs\dcs_t10_stratdyn_v1_e1350",
  [string]$TasksetSeeds = "",
  [int]$Episodes = 1350,
  [long]$EndTime = 5000000,
  [string]$ValidationSeeds = "1400:1419",
  [long]$ValidationEndTime = 5000000,
  [int]$ParallelWorkers = 3,
  [switch]$DryRun,
  [switch]$Worker,
  [int]$TasksetSeed = 0,
  [string]$WorkerStratum = "unassigned",
  [string]$WorkerStatusPath = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONPATH = "."
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$WorkloadArgs = @(
  "--workload", "mc_stratified_dynamic",
  "--mc-strat-dyn-num-tasks", "12",
  "--mc-strat-dyn-hi-ratio", "0.5",
  "--mc-strat-dyn-period-family", "seed_paired",
  "--mc-strat-dyn-period-scale", "500",
  "--mc-strat-dyn-stratum", $WorkerStratum,
  "--scenario", "stress",
  "--require-schedulable"
)
$PolicyArgs = @(
  "--dqn-runtime-semantics", "C_AMC_SEM",
  "--c-amc-sem-xf", "0.5",
  "--agent-period", "25000",
  "--reward-mode", "interval_qos_v2_single_recovery_full_C5_overinc016_abs005",
  "--action-space", "single",
  "--budget-increase-ratio", "0.02",
  "--budget-decrease-ratio", "0.02",
  "--budget-floor-ratio", "0.9",
  "--forbid-decreasing-hi-budgets",
  "--mask-detail-mode", "full",
  "--enable-deploy-cap-mask",
  "--deploy-cap-mask-ratio", "4.0",
  "--deploy-cap-mask-criticality", "lo",
  "--observation-mode", "v11_full_10d"
)

function Get-ManifestRows {
  if (-not (Test-Path -LiteralPath $Primary10Manifest)) {
    throw "primary10 manifest not found: $Primary10Manifest"
  }
  return @(Import-Csv -LiteralPath $Primary10Manifest)
}

function Invoke-WorkerBody {
  try {
    $RunDir = Join-Path $OutputRoot "tr\r0_s$TasksetSeed"
    $Arguments = @(
      "scripts/train_dqn_amc.py",
      "--episodes", $Episodes,
      "--end-time", $EndTime,
      "--validation-end-time", $ValidationEndTime,
      "--validation-seeds", $ValidationSeeds,
      "--validate-every", "10",
      "--seed", "0",
      "--fixed-taskset-seed", $TasksetSeed,
      "--output-dir", $RunDir
    ) + $WorkloadArgs + $PolicyArgs
    Write-Host "seed ${TasksetSeed}: $Python $($Arguments -join ' ')"
    if (-not $DryRun) {
      & $Python @Arguments
      if ($LASTEXITCODE -ne 0) { throw "python exit code $LASTEXITCODE" }
    }
    if ($WorkerStatusPath) {
      [System.IO.File]::WriteAllText($WorkerStatusPath, "SUCCESS")
    }
    exit 0
  } catch {
    if ($WorkerStatusPath) {
      [System.IO.File]::WriteAllText($WorkerStatusPath, "FAILED")
    }
    Write-Error "seed ${TasksetSeed}: $($_.Exception.Message)"
    exit 1
  }
}

if ($Worker) { Invoke-WorkerBody }

$Rows = Get-ManifestRows
if ([string]::IsNullOrWhiteSpace($TasksetSeeds)) {
  $Rows = @($Rows | Sort-Object { [int]$_.candidate_seed })
} else {
  $Wanted = @($TasksetSeeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
  $Rows = @($Rows | Where-Object { $Wanted -contains [string]$_.candidate_seed })
}
if ($Rows.Count -eq 0) { throw "no tasksets selected from primary10 manifest" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Entries = @()
foreach ($Row in $Rows) {
  $Seed = [int]$Row.candidate_seed
  $Stratum = [string]$Row.stratum
  $Status = Join-Path $OutputRoot "seed_${Seed}.worker_status.txt"
  if (Test-Path -LiteralPath $Status) { Remove-Item -LiteralPath $Status -Force }
  if ($DryRun) {
    Write-Host "seed ${Seed}: dry-run worker command"
    continue
  }
  $Args = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
    "-ProjectRoot", $ProjectRoot,
    "-Python", $Python,
    "-Primary10Manifest", $Primary10Manifest,
    "-OutputRoot", $OutputRoot,
    "-Episodes", $Episodes,
    "-EndTime", $EndTime,
    "-ValidationSeeds", $ValidationSeeds,
    "-ValidationEndTime", $ValidationEndTime,
    "-TasksetSeed", $Seed,
    "-WorkerStratum", $Stratum,
    "-WorkerStatusPath", $Status,
    "-Worker"
  )
  $Process = Start-Process -FilePath (Join-Path $PSHOME "powershell.exe") -ArgumentList $Args -PassThru
  $Entries += [pscustomobject]@{ Seed = $Seed; Process = $Process; Status = $Status }
  while (@($Entries | Where-Object { -not $_.Process.HasExited }).Count -ge $ParallelWorkers) {
    Start-Sleep -Milliseconds 200
  }
}

foreach ($Entry in $Entries) {
  $Entry.Process.WaitForExit()
  $StatusText = if (Test-Path -LiteralPath $Entry.Status) { (Get-Content -Raw $Entry.Status).Trim() } else { "" }
  $ExitCode = $null
  try { $ExitCode = $Entry.Process.ExitCode } catch { $ExitCode = $null }
  $Succeeded = ($StatusText -eq "SUCCESS")
  $Failed = ($StatusText -eq "FAILED")
  if ((-not $Succeeded) -and (-not $Failed) -and ($null -ne $ExitCode)) {
    $Succeeded = ([int]$ExitCode -eq 0)
  }
  if (-not $Succeeded) {
    $ExitText = if ($null -eq $ExitCode) { "<unavailable>" } else { [string]$ExitCode }
    throw "seed $($Entry.Seed) worker failed; worker_status=$StatusText; exit_code=$ExitText"
  }
  Write-Host "seed $($Entry.Seed): SUCCESS"
}
