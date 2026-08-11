param(
  [string]$ProjectRoot = ".",
  [string]$Python = "python",
  [string]$Primary10Manifest = "outputs\tasksets\mc_stratified_dynamic_v1\mc_stratified_dynamic_primary10_v1.csv",
  [string]$TrainingRoot = "outputs\dcs_t10_stratdyn_v1_e1350",
  [string]$OutputRoot = "outputs\dcs_t10_stratdyn_v1_e1350\hout\csem_stratdyn_v1",
  [string]$StaticBudgetConfig = "",
  [string]$PressureHeuristicConfig = "",
  [string]$HoutSeeds = "1550:1599",
  [long]$EndTime = 20000000,
  [int]$EvaluationWorkers = 1,
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
$env:PYTHONPATH = "."
$env:KMP_DUPLICATE_LIB_OK = "TRUE"

$WorkloadArgs = @(
  "--workload", "mc_stratified_dynamic",
  "--scenario", "stress",
  "--mc-strat-dyn-num-tasks", "12",
  "--mc-strat-dyn-hi-ratio", "0.5",
  "--mc-strat-dyn-period-family", "seed_paired",
  "--mc-strat-dyn-period-scale", "500",
  "--mc-strat-dyn-stratum", $WorkerStratum,
  "--fixed-taskset-seed", $TasksetSeed,
  "--scenario-seed-offset", "100000",
  "--require-schedulable"
)
$RuntimeArgs = @(
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

function Invoke-WorkerBody {
  try {
    $Model = Join-Path $TrainingRoot "tr\r0_s$TasksetSeed\model_best.pt"
    $Output = Join-Path $OutputRoot "r0_s$TasksetSeed\hout.csv"
    $StaticConfigForSeed = if ($StaticBudgetConfig) {
      $StaticBudgetConfig.Replace("{seed}", [string]$TasksetSeed)
    } else { "" }
    $PressureConfigForSeed = if ($PressureHeuristicConfig) {
      $PressureHeuristicConfig.Replace("{seed}", [string]$TasksetSeed)
    } else { "" }
    $BaselineMethods = @(
      "c_amc_sem_baseline",
      "amc_plus_baseline",
      "noop_agent",
      "random_valid_agent",
      "global_fixed_pressure",
      "dqn_agent"
    )
    if ($StaticConfigForSeed) { $BaselineMethods += "static_tuned_budget" }
    if ($PressureConfigForSeed) { $BaselineMethods += "pressure_threshold_valid_agent" }
    $Arguments = @(
      "scripts/evaluate_dqn_amc.py",
      "--model", $Model,
      "--seeds", $HoutSeeds,
      "--evaluation-workers", $EvaluationWorkers,
      "--end-time", $EndTime,
      "--baselines", ($BaselineMethods -join ","),
      "--output", $Output
    ) + $WorkloadArgs + $RuntimeArgs
    if ($StaticConfigForSeed) {
      $Arguments += @("--static-budget-config", $StaticConfigForSeed)
    }
    if ($PressureConfigForSeed) {
      $Arguments += @("--pressure-heuristic-config", $PressureConfigForSeed)
    }
    Write-Host "seed ${TasksetSeed}: $Python $($Arguments -join ' ')"
    if (-not $DryRun) {
      & $Python @Arguments
      if ($LASTEXITCODE -ne 0) { throw "python exit code $LASTEXITCODE" }
    }
    if ($WorkerStatusPath) { [System.IO.File]::WriteAllText($WorkerStatusPath, "SUCCESS") }
    exit 0
  } catch {
    if ($WorkerStatusPath) { [System.IO.File]::WriteAllText($WorkerStatusPath, "FAILED") }
    Write-Error "seed ${TasksetSeed}: $($_.Exception.Message)"
    exit 1
  }
}
if ($Worker) { Invoke-WorkerBody }

if (-not (Test-Path -LiteralPath $Primary10Manifest)) { throw "primary10 manifest not found: $Primary10Manifest" }
$Rows = @(Import-Csv -LiteralPath $Primary10Manifest)
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Entries = @()
foreach ($Row in $Rows) {
  $Seed = [int]$Row.candidate_seed
  $Stratum = [string]$Row.stratum
  $Status = Join-Path $OutputRoot "seed_${Seed}.worker_status.txt"
  if (Test-Path -LiteralPath $Status) { Remove-Item -LiteralPath $Status -Force }
  if ($DryRun) { Write-Host "seed ${Seed}: dry-run HOUT command"; continue }
  $Args = @(
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $PSCommandPath,
    "-ProjectRoot", $ProjectRoot, "-Python", $Python,
    "-Primary10Manifest", $Primary10Manifest, "-TrainingRoot", $TrainingRoot,
    "-OutputRoot", $OutputRoot, "-HoutSeeds", $HoutSeeds, "-EndTime", $EndTime,
    "-StaticBudgetConfig", $StaticBudgetConfig,
    "-PressureHeuristicConfig", $PressureHeuristicConfig,
    "-EvaluationWorkers", $EvaluationWorkers, "-ParallelWorkers", $ParallelWorkers,
    "-TasksetSeed", $Seed, "-WorkerStratum", $Stratum,
    "-WorkerStatusPath", $Status, "-Worker"
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
  if ((-not $Succeeded) -and (-not $Failed) -and ($null -ne $ExitCode)) { $Succeeded = ([int]$ExitCode -eq 0) }
  if (-not $Succeeded) {
    $ExitText = if ($null -eq $ExitCode) { "<unavailable>" } else { [string]$ExitCode }
    throw "seed $($Entry.Seed) worker failed; worker_status=$StatusText; exit_code=$ExitText"
  }
  Write-Host "seed $($Entry.Seed): SUCCESS"
}
