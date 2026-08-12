param(
  [string]$ProjectRoot = ".",
  [string]$Python = "python",
  [string]$Primary10Manifest = "outputs\tasksets\mc_stratified_dynamic_v1\mc_stratified_dynamic_primary10_v1.csv",
  [string]$OutputRoot = "outputs\dcs_t10_stratdyn_v1_e1350_explicit_noop_loqosdirect_v2_gamma0995",
  [string]$HoutOutputRoot = "outputs\dcs_t10_stratdyn_v1_e1350_explicit_noop_loqosdirect_v2_gamma0995\hout\all_baselines_perseed",
  [string]$TasksetSeeds = "",
  [int]$Episodes = 1350,
  [long]$EndTime = 5000000,
  [string]$ValidationSeeds = "1400:1419",
  [long]$ValidationEndTime = 5000000,
  [int]$ParallelTrainSeeds = 2,
  [double]$NoopExplorationProb = 0.04,
  [double]$Gamma = 0.995,
  [int]$CheckpointEvery = 50,
  [int]$ValidationWorkers = 6,
  [string]$HoutSeeds = "1550:1599",
  [string]$HoutEndTimes = "20000000,50000000",
  [long]$BaselineSelectionEndTime = 5000000,
  [int]$EvaluationWorkers = 1,
  [int]$ParallelHoutSeeds = 1,
  [switch]$SkipHout,
  [switch]$Force,
  [switch]$DryRun,

  # Internal worker parameters.
  [ValidateSet("", "train", "hout")]
  [string]$WorkerRole = "",
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

if ($ParallelTrainSeeds -lt 1) { throw "ParallelTrainSeeds must be >= 1" }
if ($NoopExplorationProb -lt 0.0 -or $NoopExplorationProb -gt 1.0) { throw "NoopExplorationProb must be in [0, 1]" }
if ($Gamma -le 0.0 -or $Gamma -gt 1.0) { throw "Gamma must be in (0, 1]" }
if ($CheckpointEvery -lt 1) { throw "CheckpointEvery must be >= 1" }
if ($ValidationWorkers -lt 1) { throw "ValidationWorkers must be >= 1" }
if ($ParallelHoutSeeds -lt 1) { throw "ParallelHoutSeeds must be >= 1" }
if ($EvaluationWorkers -lt 1) { throw "EvaluationWorkers must be >= 1" }

$RewardMode = "interval_lo_quality_predictive_v4_medium"
$AgentPeriod = 25000

# "Old" non-tree baselines from the pre-existing evaluator plus the four
# added C-AMC-sem baselines.  global_fixed_pressure is kept as an additional
# strong diagnostic because it was central to diagnosing the old taskset bias.
$AllBaselineMethods = @(
  "amc_plus_baseline",
  "amc_ra_baseline",
  "amc_rh_baseline",
  "c_amc_sem_baseline",
  "random_agent",
  "heuristic_agent",
  "dqn_agent",
  "noop_agent",
  "static_tuned_budget",
  "random_valid_agent",
  "pressure_threshold_valid_agent",
  "global_fixed_pressure"
)

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
  "--agent-period", [string]$AgentPeriod,
  "--reward-mode", $RewardMode,
  "--action-space", "single",
  "--include-explicit-noop",
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

$FeatureArgs = @(
  "--ema-alpha", "0.2",
  "--overrun-ema-alpha", "0.1",
  "--history-k", "8",
  "--event-window", "10",
  "--max-cost-weight", "0.7",
  "--risk-max-scale", "3.0",
  "--include-safety-margin"
)

function Parse-List {
  param([string]$Raw)
  return @($Raw.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

function Get-HoutLabel {
  param([long]$Value)
  if ($Value -eq 20000000) { return "h2" }
  if ($Value -eq 50000000) { return "h5" }
  return "t$Value"
}

function Write-JsonNoBom {
  param(
    [string]$Path,
    [object]$Payload
  )
  $Parent = Split-Path -Parent $Path
  if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
  }
  $Json = $Payload | ConvertTo-Json -Depth 8
  $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Json, $Utf8NoBom)
}

function Write-Status {
  param([string]$Text)
  if ([string]::IsNullOrWhiteSpace($WorkerStatusPath)) { return }
  $Parent = Split-Path -Parent $WorkerStatusPath
  if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
  }
  [System.IO.File]::WriteAllText($WorkerStatusPath, $Text)
}

function Invoke-PythonCommand {
  param(
    [object[]]$Arguments,
    [switch]$AllowFailure
  )

  Write-Host "$Python $($Arguments -join ' ')"
  if ($DryRun) { return $true }

  & $Python @Arguments
  $Code = $LASTEXITCODE
  if ($Code -ne 0) {
    if ($AllowFailure) {
      Write-Host "WARN: python command failed with exit code $Code" -ForegroundColor Yellow
      return $false
    }
    throw "python command failed with exit code ${Code}: $($Arguments -join ' ')"
  }
  return $true
}

function Resolve-ModelBestPath {
  param([string]$RunDir)
  $Model = Join-Path $RunDir "model_best_lo_quality_qos_best.pt"
  if (-not (Test-Path -LiteralPath $Model)) {
    throw "Expected best-model checkpoint not found: $Model"
  }
  return $Model
}

function Get-Mean {
  param(
    [object[]]$Rows,
    [string]$Column
  )
  if ($Rows.Count -eq 0) { return [double]::NaN }
  $Sum = 0.0
  foreach ($Row in $Rows) {
    $Sum += [double]$Row.$Column
  }
  return $Sum / [double]$Rows.Count
}

function Invoke-Evaluation {
  param(
    [string]$Model,
    [string]$Seeds,
    [long]$EvalEndTime,
    [string[]]$Methods,
    [string]$OutputCsv,
    [string]$StaticConfig = "",
    [string]$PressureConfig = "",
    [switch]$AllowFailure
  )

  $Parent = Split-Path -Parent $OutputCsv
  if ($Parent -and -not (Test-Path -LiteralPath $Parent)) {
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
  }
  if ((Test-Path -LiteralPath $OutputCsv) -and $Force) {
    Remove-Item -LiteralPath $OutputCsv -Force
  }

  $Args = @(
    "scripts/evaluate_dqn_amc.py",
    "--model", $Model,
    "--seeds", $Seeds,
    "--evaluation-workers", [string]$EvaluationWorkers,
    "--end-time", [string]$EvalEndTime,
    "--baselines", ($Methods -join ","),
    "--random-valid-policy-seed-offset", "0",
    "--max-q-diagnostic-samples", "1000",
    "--output", $OutputCsv
  ) + $WorkloadArgs + $RuntimeArgs + $FeatureArgs

  if (-not [string]::IsNullOrWhiteSpace($StaticConfig)) {
    $Args += @("--static-budget-config", $StaticConfig)
  }
  if (-not [string]::IsNullOrWhiteSpace($PressureConfig)) {
    $Args += @("--pressure-heuristic-config", $PressureConfig)
  }

  return Invoke-PythonCommand -Arguments $Args -AllowFailure:$AllowFailure
}

function Select-StaticBaselineForSeed {
  param(
    [string]$Model,
    [string]$SelectionDir
  )

  $FinalConfig = Join-Path $SelectionDir "selected_static_budget.json"
  if ((Test-Path -LiteralPath $FinalConfig) -and (-not $Force)) {
    Write-Host "seed ${TasksetSeed}: reuse static baseline selection"
    return $FinalConfig
  }

  New-Item -ItemType Directory -Force -Path $SelectionDir | Out-Null
  $AlphaGrid = @(1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00)
  $CandidateRows = @()

  foreach ($Alpha in $AlphaGrid) {
    $Tag = ([string]$Alpha).Replace(".", "p")
    $ConfigPath = Join-Path $SelectionDir "candidate_alpha_$Tag.json"
    $CsvPath = Join-Path $SelectionDir "candidate_alpha_$Tag.csv"

    $Payload = [ordered]@{
      taskset_seed = $TasksetSeed
      selection_seeds = $ValidationSeeds
      selection_end_time = $BaselineSelectionEndTime
      alpha = [double]$Alpha
      selection_metric = "lo_quality_qos"
      selection_metric_value = 0.0
      validation_lo_zero_service_ratio = 0.0
      validation_lo_equiv_jne = 0.0
      budgets = @{}
    }
    Write-JsonNoBom -Path $ConfigPath -Payload $Payload

    $Ok = Invoke-Evaluation `
      -Model $Model `
      -Seeds $ValidationSeeds `
      -EvalEndTime $BaselineSelectionEndTime `
      -Methods @("static_tuned_budget") `
      -OutputCsv $CsvPath `
      -StaticConfig $ConfigPath `
      -AllowFailure

    if (-not $Ok -or -not (Test-Path -LiteralPath $CsvPath)) {
      $CandidateRows += [pscustomobject]@{
        alpha = [double]$Alpha
        feasible = $false
        qos = [double]::NegativeInfinity
        zero = [double]::PositiveInfinity
        jne = [double]::PositiveInfinity
        misses = [int]::MaxValue
      }
      continue
    }

    $Rows = @(Import-Csv -LiteralPath $CsvPath)
    $Misses = [int](($Rows | Measure-Object -Property deadline_misses -Sum).Sum)
    $Qos = Get-Mean -Rows $Rows -Column "lo_quality_qos"
    $Zero = Get-Mean -Rows $Rows -Column "lo_zero_service_ratio"
    $Jne = Get-Mean -Rows $Rows -Column "lo_equiv_jne"

    $CandidateRows += [pscustomobject]@{
      alpha = [double]$Alpha
      feasible = ($Misses -eq 0)
      qos = [double]$Qos
      zero = [double]$Zero
      jne = [double]$Jne
      misses = [int]$Misses
    }
  }

  $CandidateRows | Export-Csv -LiteralPath (Join-Path $SelectionDir "static_candidates.csv") -NoTypeInformation

  $Feasible = @($CandidateRows | Where-Object { $_.feasible })
  if ($Feasible.Count -eq 0) {
    throw "seed ${TasksetSeed}: no feasible static baseline candidate"
  }

  $Best = $Feasible | Sort-Object `
    @{Expression = "qos"; Descending = $true}, `
    @{Expression = "zero"; Descending = $false}, `
    @{Expression = "jne"; Descending = $false}, `
    @{Expression = "alpha"; Descending = $false} | Select-Object -First 1

  $FinalPayload = [ordered]@{
    taskset_seed = $TasksetSeed
    selection_seeds = $ValidationSeeds
    selection_end_time = $BaselineSelectionEndTime
    alpha = [double]$Best.alpha
    selection_metric = "lo_quality_qos"
    selection_metric_value = [double]$Best.qos
    validation_lo_zero_service_ratio = [double]$Best.zero
    validation_lo_equiv_jne = [double]$Best.jne
    budgets = @{}
  }
  Write-JsonNoBom -Path $FinalConfig -Payload $FinalPayload
  Write-Host "seed ${TasksetSeed}: selected static alpha=$($Best.alpha), qos=$($Best.qos)"
  return $FinalConfig
}

function Select-PressureBaselineForSeed {
  param(
    [string]$Model,
    [string]$SelectionDir
  )

  $FinalConfig = Join-Path $SelectionDir "selected_pressure_heuristic.json"
  if ((Test-Path -LiteralPath $FinalConfig) -and (-not $Force)) {
    Write-Host "seed ${TasksetSeed}: reuse pressure baseline selection"
    return $FinalConfig
  }

  New-Item -ItemType Directory -Force -Path $SelectionDir | Out-Null
  $ULows = @(0.80, 0.90, 1.00)
  $UHighs = @(1.00, 1.10, 1.20)
  $CandidateRows = @()

  foreach ($ULow in $ULows) {
    foreach ($UHigh in $UHighs) {
      if ($ULow -ge $UHigh) { continue }

      $LowTag = ([string]$ULow).Replace(".", "p")
      $HighTag = ([string]$UHigh).Replace(".", "p")
      $ConfigPath = Join-Path $SelectionDir "candidate_${LowTag}_${HighTag}.json"
      $CsvPath = Join-Path $SelectionDir "candidate_${LowTag}_${HighTag}.csv"

      $Payload = [ordered]@{
        taskset_seed = $TasksetSeed
        u_low = [double]$ULow
        u_high = [double]$UHigh
        selection_seeds = $ValidationSeeds
        selection_end_time = $BaselineSelectionEndTime
      }
      Write-JsonNoBom -Path $ConfigPath -Payload $Payload

      $Ok = Invoke-Evaluation `
        -Model $Model `
        -Seeds $ValidationSeeds `
        -EvalEndTime $BaselineSelectionEndTime `
        -Methods @("pressure_threshold_valid_agent") `
        -OutputCsv $CsvPath `
        -PressureConfig $ConfigPath `
        -AllowFailure

      if (-not $Ok -or -not (Test-Path -LiteralPath $CsvPath)) {
        continue
      }

      $Rows = @(Import-Csv -LiteralPath $CsvPath)
      $Misses = [int](($Rows | Measure-Object -Property deadline_misses -Sum).Sum)
      $Invalid = [int](($Rows | Measure-Object -Property selected_invalid_mask_actions -Sum).Sum)
      $Qos = Get-Mean -Rows $Rows -Column "lo_quality_qos"
      $Zero = Get-Mean -Rows $Rows -Column "lo_zero_service_ratio"
      $Jne = Get-Mean -Rows $Rows -Column "lo_equiv_jne"
      $ActionRate = Get-Mean -Rows $Rows -Column "accepted_action_rate"

      $CandidateRows += [pscustomobject]@{
        u_low = [double]$ULow
        u_high = [double]$UHigh
        valid = ($Misses -eq 0 -and $Invalid -eq 0)
        qos = [double]$Qos
        zero = [double]$Zero
        jne = [double]$Jne
        action_rate = [double]$ActionRate
        misses = [int]$Misses
        invalid = [int]$Invalid
        band = [double]($UHigh - $ULow)
      }
    }
  }

  if ($CandidateRows.Count -eq 0) {
    throw "seed ${TasksetSeed}: pressure baseline selection produced no candidate rows"
  }
  $CandidateRows | Export-Csv -LiteralPath (Join-Path $SelectionDir "pressure_candidates.csv") -NoTypeInformation

  $Valid = @($CandidateRows | Where-Object { $_.valid })
  if ($Valid.Count -eq 0) {
    throw "seed ${TasksetSeed}: no valid pressure baseline candidate"
  }

  $Best = $Valid | Sort-Object `
    @{Expression = "qos"; Descending = $true}, `
    @{Expression = "zero"; Descending = $false}, `
    @{Expression = "jne"; Descending = $false}, `
    @{Expression = "action_rate"; Descending = $false}, `
    @{Expression = "band"; Descending = $true}, `
    @{Expression = "u_low"; Descending = $false}, `
    @{Expression = "u_high"; Descending = $true} | Select-Object -First 1

  $FinalPayload = [ordered]@{
    taskset_seed = $TasksetSeed
    u_low = [double]$Best.u_low
    u_high = [double]$Best.u_high
    selection_seeds = $ValidationSeeds
    selection_end_time = $BaselineSelectionEndTime
    selection_metric = "lo_quality_qos"
    selection_metric_value = [double]$Best.qos
    validation_lo_zero_service_ratio = [double]$Best.zero
    validation_lo_equiv_jne = [double]$Best.jne
    validation_action_rate = [double]$Best.action_rate
  }
  Write-JsonNoBom -Path $FinalConfig -Payload $FinalPayload
  Write-Host "seed ${TasksetSeed}: selected pressure u_low=$($Best.u_low), u_high=$($Best.u_high), qos=$($Best.qos)"
  return $FinalConfig
}

function Invoke-TrainWorker {
  try {
    $RunDir = Join-Path $OutputRoot "tr\r0_s$TasksetSeed"
    $BestQoS = Join-Path $RunDir "model_best_lo_quality_qos_best.pt"

    if ((-not $Force) -and (Test-Path -LiteralPath $BestQoS)) {
      Write-Host "seed ${TasksetSeed}: training checkpoint already exists; skip training"
      Write-Status -Text "SUCCESS"
      exit 0
    }

    $Arguments = @(
      "scripts/train_dqn_amc.py",
      "--episodes", [string]$Episodes,
      "--end-time", [string]$EndTime,
      "--validation-end-time", [string]$ValidationEndTime,
      "--validation-seeds", $ValidationSeeds,
      "--validate-every", "10",
      "--validation-workers", [string]$ValidationWorkers,
      "--save-best-by", "lo_quality_qos_best",
      "--gamma", [string]$Gamma,
      "--checkpoint", [string]$CheckpointEvery,
      "--seed", "0",
      "--fixed-taskset-seed", [string]$TasksetSeed,
      "--output-dir", $RunDir,
      "--noop-exploration-prob", [string]$NoopExplorationProb
    ) + $WorkloadArgs + $RuntimeArgs + $FeatureArgs

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "TRAIN seed=$TasksetSeed stratum=$WorkerStratum"
    Write-Host "============================================================"
    Invoke-PythonCommand -Arguments $Arguments

    if (-not $DryRun) {
      [void](Resolve-ModelBestPath -RunDir $RunDir)
    }
    Write-Status -Text "SUCCESS"
    exit 0
  } catch {
    Write-Status -Text "FAILED"
    Write-Error "TRAIN seed ${TasksetSeed}: $($_.Exception.Message)"
    exit 1
  }
}

function Invoke-HoutWorker {
  try {
    if ($SkipHout) {
      Write-Status -Text "SUCCESS"
      exit 0
    }

    $RunDir = Join-Path $OutputRoot "tr\r0_s$TasksetSeed"
    $Model = Resolve-ModelBestPath -RunDir $RunDir
    $SeedRoot = Join-Path $HoutOutputRoot "r0_s$TasksetSeed"
    $StaticDir = Join-Path $SeedRoot "static_selection"
    $PressureDir = Join-Path $SeedRoot "pressure_selection"
    New-Item -ItemType Directory -Force -Path $SeedRoot | Out-Null

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "HOUT PIPELINE seed=$TasksetSeed stratum=$WorkerStratum"
    Write-Host "============================================================"

    # Do validation tuning using the *same mc_stratified_dynamic workload*.
    # The older selector scripts are intentionally not used because they are
    # formal10/mc_fairgen-specific in the current repository.
    $StaticConfig = Select-StaticBaselineForSeed -Model $Model -SelectionDir $StaticDir
    $PressureConfig = Select-PressureBaselineForSeed -Model $Model -SelectionDir $PressureDir

    foreach ($EndText in (Parse-List -Raw $HoutEndTimes)) {
      $EvalEndTime = [long]$EndText
      $Label = Get-HoutLabel -Value $EvalEndTime
      $OutputCsv = Join-Path $SeedRoot "hout_$Label.csv"

      if ((Test-Path -LiteralPath $OutputCsv) -and (-not $Force)) {
        Write-Host "seed ${TasksetSeed}: reuse existing HOUT $Label"
        continue
      }

      Write-Host "seed ${TasksetSeed}: HOUT $Label, methods=$($AllBaselineMethods -join ',')"
      Invoke-Evaluation `
        -Model $Model `
        -Seeds $HoutSeeds `
        -EvalEndTime $EvalEndTime `
        -Methods $AllBaselineMethods `
        -OutputCsv $OutputCsv `
        -StaticConfig $StaticConfig `
        -PressureConfig $PressureConfig | Out-Null
    }

    Write-Status -Text "SUCCESS"
    exit 0
  } catch {
    Write-Status -Text "FAILED"
    Write-Error "HOUT seed ${TasksetSeed}: $($_.Exception.Message)"
    exit 1
  }
}

if ($WorkerRole -eq "train") { Invoke-TrainWorker }
if ($WorkerRole -eq "hout") { Invoke-HoutWorker }

function Get-ManifestRows {
  if (-not (Test-Path -LiteralPath $Primary10Manifest)) {
    throw "primary10 manifest not found: $Primary10Manifest"
  }

  $Rows = @(Import-Csv -LiteralPath $Primary10Manifest)
  if ([string]::IsNullOrWhiteSpace($TasksetSeeds)) {
    return @($Rows | Sort-Object { [int]$_.candidate_seed })
  }

  $Wanted = @($TasksetSeeds.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ })
  return @($Rows | Where-Object { $Wanted -contains [string]$_.candidate_seed } | Sort-Object { [int]$_.candidate_seed })
}

function Start-SeedWorker {
  param(
    [ValidateSet("train", "hout")]
    [string]$Role,
    [int]$Seed,
    [string]$Stratum,
    [string]$StatusPath
  )

  if (Test-Path -LiteralPath $StatusPath) {
    Remove-Item -LiteralPath $StatusPath -Force
  }

  $Args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $PSCommandPath,
    "-ProjectRoot", $ProjectRoot,
    "-Python", $Python,
    "-Primary10Manifest", $Primary10Manifest,
    "-OutputRoot", $OutputRoot,
    "-HoutOutputRoot", $HoutOutputRoot,
    "-Episodes", [string]$Episodes,
    "-EndTime", [string]$EndTime,
    "-ValidationSeeds", $ValidationSeeds,
    "-ValidationEndTime", [string]$ValidationEndTime,
    "-ParallelTrainSeeds", "1",
    "-NoopExplorationProb", [string]$NoopExplorationProb,
    "-Gamma", [string]$Gamma,
    "-CheckpointEvery", [string]$CheckpointEvery,
    "-ValidationWorkers", [string]$ValidationWorkers,
    "-HoutSeeds", $HoutSeeds,
    "-HoutEndTimes", $HoutEndTimes,
    "-BaselineSelectionEndTime", [string]$BaselineSelectionEndTime,
    "-EvaluationWorkers", [string]$EvaluationWorkers,
    "-ParallelHoutSeeds", "1",
    "-TasksetSeed", [string]$Seed,
    "-WorkerStratum", $Stratum,
    "-WorkerStatusPath", $StatusPath,
    "-WorkerRole", $Role
  )

  if ($Force) { $Args += "-Force" }
  if ($DryRun) { $Args += "-DryRun" }
  if ($SkipHout) { $Args += "-SkipHout" }

  if ($DryRun) {
    Write-Host "DRY-RUN $Role seed=$Seed"
    Write-Host "powershell.exe $($Args -join ' ')"
    return $null
  }

  $PowerShellExe = Join-Path $PSHOME "powershell.exe"
  return Start-Process -FilePath $PowerShellExe -ArgumentList $Args -PassThru
}

function Get-WorkerSucceeded {
  param(
    [object]$Entry
  )
  $Entry.Process.WaitForExit()
  $StatusText = if (Test-Path -LiteralPath $Entry.Status) {
    (Get-Content -Raw -LiteralPath $Entry.Status).Trim()
  } else {
    ""
  }

  $ExitCode = $null
  try { $ExitCode = $Entry.Process.ExitCode } catch { $ExitCode = $null }

  if ($StatusText -eq "SUCCESS") { return $true }
  if ($StatusText -eq "FAILED") { return $false }
  if ($null -ne $ExitCode) { return ([int]$ExitCode -eq 0) }
  return $false
}

$Rows = Get-ManifestRows
if ($Rows.Count -eq 0) { throw "no tasksets selected from primary10 manifest" }

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
if (-not $SkipHout) {
  New-Item -ItemType Directory -Force -Path $HoutOutputRoot | Out-Null
}

if ($DryRun) {
  foreach ($Row in $Rows) {
    $Seed = [int]$Row.candidate_seed
    $Stratum = [string]$Row.stratum
    $TrainStatus = Join-Path $OutputRoot "seed_${Seed}.train_status.txt"
    $null = Start-SeedWorker -Role "train" -Seed $Seed -Stratum $Stratum -StatusPath $TrainStatus
    if (-not $SkipHout) {
      $HoutStatus = Join-Path $HoutOutputRoot "seed_${Seed}.hout_status.txt"
      $null = Start-SeedWorker -Role "hout" -Seed $Seed -Stratum $Stratum -StatusPath $HoutStatus
    }
  }
  exit 0
}

# Scheduler:
# - up to ParallelTrainSeeds training workers are kept busy;
# - every successful training immediately enters the HOUT queue;
# - HOUT has its own concurrency limit and never consumes a training slot.
$NextTrainIndex = 0
$ActiveTrain = @()
$HoutQueue = New-Object System.Collections.Queue
$ActiveHout = @()
$Failures = @()

while (
  $NextTrainIndex -lt $Rows.Count -or
  $ActiveTrain.Count -gt 0 -or
  $HoutQueue.Count -gt 0 -or
  $ActiveHout.Count -gt 0
) {
  while ($NextTrainIndex -lt $Rows.Count -and $ActiveTrain.Count -lt $ParallelTrainSeeds) {
    $Row = $Rows[$NextTrainIndex]
    $NextTrainIndex += 1

    $Seed = [int]$Row.candidate_seed
    $Stratum = [string]$Row.stratum
    $Status = Join-Path $OutputRoot "seed_${Seed}.train_status.txt"
    Write-Host "LAUNCH TRAIN seed=$Seed stratum=$Stratum"
    $Process = Start-SeedWorker -Role "train" -Seed $Seed -Stratum $Stratum -StatusPath $Status
    $ActiveTrain += [pscustomobject]@{
      Seed = $Seed
      Stratum = $Stratum
      Process = $Process
      Status = $Status
    }
  }

  $StillTraining = @()
  foreach ($Entry in $ActiveTrain) {
    if (-not $Entry.Process.HasExited) {
      $StillTraining += $Entry
      continue
    }

    if (Get-WorkerSucceeded -Entry $Entry) {
      Write-Host "TRAIN SUCCESS seed=$($Entry.Seed)" -ForegroundColor Green
      if (-not $SkipHout) {
        $HoutQueue.Enqueue([pscustomobject]@{
          Seed = $Entry.Seed
          Stratum = $Entry.Stratum
        })
      }
    } else {
      Write-Host "TRAIN FAILED seed=$($Entry.Seed)" -ForegroundColor Red
      $Failures += "train:$($Entry.Seed)"
    }
  }
  $ActiveTrain = $StillTraining

  while ($HoutQueue.Count -gt 0 -and $ActiveHout.Count -lt $ParallelHoutSeeds) {
    $Item = $HoutQueue.Dequeue()
    $Status = Join-Path $HoutOutputRoot "seed_$($Item.Seed).hout_status.txt"
    Write-Host "LAUNCH HOUT seed=$($Item.Seed) stratum=$($Item.Stratum)"
    $Process = Start-SeedWorker -Role "hout" -Seed $Item.Seed -Stratum $Item.Stratum -StatusPath $Status
    $ActiveHout += [pscustomobject]@{
      Seed = $Item.Seed
      Stratum = $Item.Stratum
      Process = $Process
      Status = $Status
    }
  }

  $StillHout = @()
  foreach ($Entry in $ActiveHout) {
    if (-not $Entry.Process.HasExited) {
      $StillHout += $Entry
      continue
    }

    if (Get-WorkerSucceeded -Entry $Entry) {
      Write-Host "HOUT SUCCESS seed=$($Entry.Seed)" -ForegroundColor Green
    } else {
      Write-Host "HOUT FAILED seed=$($Entry.Seed)" -ForegroundColor Red
      $Failures += "hout:$($Entry.Seed)"
    }
  }
  $ActiveHout = $StillHout

  if (
    $NextTrainIndex -lt $Rows.Count -or
    $ActiveTrain.Count -gt 0 -or
    $HoutQueue.Count -gt 0 -or
    $ActiveHout.Count -gt 0
  ) {
    Start-Sleep -Milliseconds 500
  }
}

if ($Failures.Count -gt 0) {
  throw "One or more seed pipelines failed: $($Failures -join ', ')"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "ALL SELECTED SEEDS COMPLETED"
Write-Host "training root: $OutputRoot"
if (-not $SkipHout) {
  Write-Host "HOUT root: $HoutOutputRoot"
  Write-Host "methods: $($AllBaselineMethods -join ',')"
  Write-Host "reward mode: $RewardMode; best selector: lo_quality_qos_best"
  Write-Host "explicit noop: enabled; epsilon noop exploration probability=$NoopExplorationProb; validation workers=$ValidationWorkers"
}
Write-Host "============================================================"
