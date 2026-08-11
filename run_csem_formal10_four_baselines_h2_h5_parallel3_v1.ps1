param(
  [string]$ProjectRoot = "D:\AMC",
  [string]$CondaEnv = "amc-repro-macosmatch",
  [string]$TrainingOutputRoot = "outputs\dcs_t10_e1350",
  [string]$FormalRoot = "",
  [string]$TasksetSeeds = "2221,397,861,639,1264,1502,358,185,2535,2829",
  [string]$ValidationSeeds = "1400:1419",
  [string]$HoutSeeds = "1550:1599",
  [long]$ValidationEndTime = 20000000,
  [string]$HoutEndTimes = "20000000,50000000",
  [int]$AgentPeriod = 25000,
  [int]$ParallelTasksets = 3,
  [int]$EvaluationWorkers = 1,
  [switch]$Force,
  [switch]$SkipHout,
  [switch]$ContinueOnSeedFailure,
  [switch]$SeedWorker
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location $ProjectRoot

if ([string]::IsNullOrWhiteSpace($FormalRoot)) {
  $FormalRoot = Join-Path $ProjectRoot "outputs\dcs_t10_e1350\csem_formal10_four_baselines_s1550_1599_h2_h5"
}
$TrainingRoot = Join-Path $ProjectRoot $TrainingOutputRoot
$GlobalFailureLog = Join-Path $FormalRoot "formal10_baseline_failures.txt"

$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTHONPATH = "."
$env:PYTHONHASHSEED = "0"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$RewardMode = "interval_qos_v2_single_recovery_full_C5_overinc016_abs005"
$DqnRuntimeSemantics = "C_AMC_SEM"
$CAmcSemXf = 0.5
$BaselineMethods = "c_amc_sem_baseline,noop_agent,static_tuned_budget,random_valid_agent,pressure_threshold_valid_agent"

function Assert-Exists {
  param([string]$Path, [string]$Message)
  if (!(Test-Path $Path)) { throw "$Message Path=$Path" }
}

function Parse-List {
  param([string]$Raw)
  return @($Raw.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" })
}

function Write-TextNoBom {
  param([string]$Path, [string]$Text)
  $Parent = Split-Path -Parent $Path
  if ($Parent -and !(Test-Path $Parent)) {
    New-Item -ItemType Directory -Force -Path $Parent | Out-Null
  }
  $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($Path, $Text, $Utf8NoBom)
}

function Invoke-CondaPython {
  param(
    [object[]]$Arguments,
    [switch]$AllowFailure
  )

  $TempStdout = [System.IO.Path]::GetTempFileName()
  $TempStderr = [System.IO.Path]::GetTempFileName()
  $OldEap = $ErrorActionPreference
  $HasNativePref = $false
  $OldNativePref = $null
  try {
    $ErrorActionPreference = "Continue"
    $NativePrefVar = Get-Variable -Name PSNativeCommandUseErrorActionPreference -Scope Global -ErrorAction SilentlyContinue
    if ($null -ne $NativePrefVar) {
      $HasNativePref = $true
      $OldNativePref = $global:PSNativeCommandUseErrorActionPreference
      $global:PSNativeCommandUseErrorActionPreference = $false
    }

    $ArgList = @("run", "--no-capture-output", "-n", $CondaEnv) + $Arguments
    & conda @ArgList 1> $TempStdout 2> $TempStderr
    $Code = $LASTEXITCODE
    if (Test-Path $TempStdout) {
      Get-Content $TempStdout | ForEach-Object { Write-Host $_ }
    }
    if (Test-Path $TempStderr) {
      Get-Content $TempStderr | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    }
    if ($Code -ne 0) {
      if ($AllowFailure) {
        Write-Host "WARN: command failed with exit code ${Code}: $($Arguments -join ' ')" -ForegroundColor Yellow
        return $false
      }
      throw "Command failed with exit code ${Code}: $($Arguments -join ' ')"
    }
    return $true
  }
  finally {
    $ErrorActionPreference = $OldEap
    if ($HasNativePref) { $global:PSNativeCommandUseErrorActionPreference = $OldNativePref }
    if (Test-Path $TempStdout) { Remove-Item -Force $TempStdout -ErrorAction SilentlyContinue }
    if (Test-Path $TempStderr) { Remove-Item -Force $TempStderr -ErrorAction SilentlyContinue }
  }
}

function Resolve-ModelBestPath {
  param([string]$RunDir)
  foreach ($Candidate in @(
    (Join-Path $RunDir "model_best_qos_recovery_stable.pt"),
    (Join-Path $RunDir "model_best.pt")
  )) {
    if (Test-Path $Candidate) { return $Candidate }
  }
  throw "No model-best checkpoint found under $RunDir. This runner refuses model_final.pt."
}

function Assert-CsvHasColumns {
  param([string]$CsvPath, [string[]]$Columns)
  Assert-Exists -Path $CsvPath -Message "Expected CSV missing."
  $Header = (Get-Content -Path $CsvPath -TotalCount 1).Split(",")
  $Missing = @($Columns | Where-Object { $Header -notcontains $_ })
  if ($Missing.Count -gt 0) {
    throw "CSV is missing required columns: $($Missing -join ', ') ; file=$CsvPath"
  }
}

function ConvertTo-PsSingleQuotedLiteral {
  param([string]$Value)
  return "'" + ($Value -replace "'", "''") + "'"
}

function Get-HoutLabel {
  param([long]$EndTime)
  if ($EndTime -eq 20000000) { return "h2" }
  if ($EndTime -eq 50000000) { return "h5" }
  return "t$EndTime"
}

function Get-CommonWorkloadArgs {
  param([int]$TasksetSeed)
  return @(
    "--workload", "mc_fairgen",
    "--scenario", "stress",
    "--mc-fairgen-mode", "paper_learnable_headroom",
    "--mc-fairgen-num-tasks", 12,
    "--mc-fairgen-hi-ratio", 0.5,
    "--mc-fairgen-period-source", "controlled_medium",
    "--mc-fairgen-period-scale", 500,
    "--mc-fairgen-u-hi-lo-min", 0.20,
    "--mc-fairgen-u-hi-lo-max", 0.35,
    "--mc-fairgen-u-hi-hi-min", 0.45,
    "--mc-fairgen-u-hi-hi-max", 0.70,
    "--mc-fairgen-u-lo-lo-min", 0.25,
    "--mc-fairgen-u-lo-lo-max", 0.45,
    "--mc-fairgen-hi-budget-rho-min", 0.55,
    "--mc-fairgen-hi-budget-rho-max", 0.75,
    "--mc-fairgen-lo-budget-rho-min", 0.20,
    "--mc-fairgen-lo-budget-rho-max", 0.40,
    "--mc-fairgen-hi-overrun-prob", 0.08,
    "--mc-fairgen-lo-overrun-prob", 0.12,
    "--mc-fairgen-hi-overrun-factor-min", 1.02,
    "--mc-fairgen-hi-overrun-factor-max", 1.25,
    "--mc-fairgen-lo-overrun-factor-min", 1.02,
    "--mc-fairgen-lo-overrun-factor-max", 1.25,
    "--fixed-taskset-seed", $TasksetSeed,
    "--scenario-seed-offset", 100000,
    "--require-schedulable"
  )
}

$CommonRuntimeArgs = @(
  "--dqn-runtime-semantics", $DqnRuntimeSemantics,
  "--c-amc-sem-xf", $CAmcSemXf,
  "--reward-mode", $RewardMode,
  "--action-space", "single",
  "--budget-increase-ratio", 0.02,
  "--budget-decrease-ratio", 0.02,
  "--budget-floor-ratio", 0.9,
  "--forbid-decreasing-hi-budgets",
  "--mask-detail-mode", "full",
  "--enable-deploy-cap-mask",
  "--deploy-cap-mask-ratio", 4.0,
  "--deploy-cap-mask-criticality", "lo"
)

$FeatureArgs = @(
  "--observation-mode", "v11_full_10d",
  "--ema-alpha", 0.2,
  "--overrun-ema-alpha", 0.1,
  "--history-k", 8,
  "--event-window", 10,
  "--max-cost-weight", 0.7,
  "--risk-max-scale", 3.0,
  "--include-safety-margin"
)

function Invoke-TasksetBaselineSuite {
  param([int]$TasksetSeed)

  $SeedRoot = Join-Path $FormalRoot "r0_s$TasksetSeed"
  $BaselineRoot = Join-Path $SeedRoot "baseline_suite"
  $StaticDir = Join-Path $BaselineRoot "static_selection"
  $HeuristicDir = Join-Path $BaselineRoot "heuristic_selection"
  $StaticConfig = Join-Path $StaticDir "selected_static_budget.json"
  $HeuristicConfig = Join-Path $HeuristicDir "selected_pressure_heuristic.json"
  $TeacherModel = Resolve-ModelBestPath -RunDir (Join-Path $TrainingRoot "tr\r0_s$TasksetSeed")

  New-Item -ItemType Directory -Force -Path $BaselineRoot | Out-Null
  Write-Host ""
  Write-Host "============================================================"
  Write-Host "C-AMC-sem four-baseline formal10 taskset $TasksetSeed"
  Write-Host "============================================================"

  # Fixed per-taskset order: static selection, pressure selection, h2 HOUT, h5 HOUT.
  if ($Force -or !(Test-Path $StaticConfig)) {
    Invoke-CondaPython -Arguments @(
      "python", "-u", "scripts\select_csem_static_budget_baseline.py",
      "--taskset-seed", $TasksetSeed,
      "--validation-seeds", $ValidationSeeds,
      "--end-time", $ValidationEndTime,
      "--c-amc-sem-xf", $CAmcSemXf,
      "--output-dir", $StaticDir
    ) | Out-Null
  } else {
    Write-Host "SKIP existing static selection: $StaticConfig"
  }
  Assert-Exists -Path $StaticConfig -Message "Static selection did not produce selected_static_budget.json."

  if ($Force -or !(Test-Path $HeuristicConfig)) {
    Invoke-CondaPython -Arguments @(
      "python", "-u", "scripts\select_csem_pressure_heuristic.py",
      "--taskset-seed", $TasksetSeed,
      "--validation-seeds", $ValidationSeeds,
      "--end-time", $ValidationEndTime,
      "--output-dir", $HeuristicDir
    ) | Out-Null
  } else {
    Write-Host "SKIP existing pressure selection: $HeuristicConfig"
  }
  Assert-Exists -Path $HeuristicConfig -Message "Pressure selection did not produce selected_pressure_heuristic.json."

  if ($SkipHout) { return }
  $WorkloadArgs = Get-CommonWorkloadArgs -TasksetSeed $TasksetSeed
  foreach ($EndTimeText in (Parse-List -Raw $HoutEndTimes)) {
    $EndTime = [long]$EndTimeText
    $Label = Get-HoutLabel -EndTime $EndTime
    $HoutCsv = Join-Path $BaselineRoot "hout_$Label.csv"
    if ((Test-Path $HoutCsv) -and (-not $Force)) {
      Write-Host "SKIP existing $Label HOUT: $HoutCsv"
      continue
    }
    if (Test-Path $HoutCsv) { Remove-Item -Force $HoutCsv }

    Write-Host "HOUT $($Label): taskset=$TasksetSeed scenarios=$HoutSeeds end_time=$EndTime"
    $EvalArgs = @(
      "python", "-u", "scripts\evaluate_dqn_amc.py",
      "--model", $TeacherModel,
      "--seeds", $HoutSeeds,
      "--evaluation-workers", $EvaluationWorkers,
      "--end-time", $EndTime,
      "--agent-period", $AgentPeriod,
      "--baselines", $BaselineMethods,
      "--static-budget-config", $StaticConfig,
      "--pressure-heuristic-config", $HeuristicConfig,
      "--random-valid-policy-seed-offset", 0,
      "--double-dqn",
      "--max-q-diagnostic-samples", 1000,
      "--output", $HoutCsv
    ) + $WorkloadArgs + $CommonRuntimeArgs + $FeatureArgs
    Invoke-CondaPython -Arguments $EvalArgs | Out-Null
    Assert-CsvHasColumns -CsvPath $HoutCsv -Columns @(
      "method", "taskset_seed", "seed", "scenario_seed",
      "lo_quality_qos", "lo_zero_service_ratio", "lo_equiv_jne", "tid_ratio",
      "deadline_misses", "hi_deadline_misses", "lo_deadline_misses",
      "accepted_action_rate", "noop_action_rate", "selected_invalid_mask_actions",
      "masked_deploy_cap_increase_rate", "static_budget_alpha",
      "pressure_u_low", "pressure_u_high", "policy_rng_seed"
    )
  }
}

function New-SeedWorkerCommand {
  param(
    [string]$Seed,
    [string]$StatusPath
  )
  if ([string]::IsNullOrWhiteSpace($PSCommandPath)) {
    throw "Cannot resolve current script path for parallel workers."
  }
  $Parts = @(
    "& " + (ConvertTo-PsSingleQuotedLiteral -Value $PSCommandPath),
    "-ProjectRoot " + (ConvertTo-PsSingleQuotedLiteral -Value $ProjectRoot),
    "-CondaEnv " + (ConvertTo-PsSingleQuotedLiteral -Value $CondaEnv),
    "-TrainingOutputRoot " + (ConvertTo-PsSingleQuotedLiteral -Value $TrainingOutputRoot),
    "-FormalRoot " + (ConvertTo-PsSingleQuotedLiteral -Value $FormalRoot),
    "-TasksetSeeds " + (ConvertTo-PsSingleQuotedLiteral -Value $Seed),
    "-ValidationSeeds " + (ConvertTo-PsSingleQuotedLiteral -Value $ValidationSeeds),
    "-HoutSeeds " + (ConvertTo-PsSingleQuotedLiteral -Value $HoutSeeds),
    "-ValidationEndTime $ValidationEndTime",
    "-HoutEndTimes " + (ConvertTo-PsSingleQuotedLiteral -Value $HoutEndTimes),
    "-AgentPeriod $AgentPeriod",
    "-ParallelTasksets 1",
    "-EvaluationWorkers $EvaluationWorkers",
    "-SeedWorker"
  )
  if ($Force) { $Parts += "-Force" }
  if ($SkipHout) { $Parts += "-SkipHout" }
  if ($ContinueOnSeedFailure) { $Parts += "-ContinueOnSeedFailure" }
  $Invocation = $Parts -join " "
  $StatusLiteral = ConvertTo-PsSingleQuotedLiteral -Value $StatusPath
  return @"
`$ErrorActionPreference = 'Stop'
try {
  $Invocation
  [System.IO.File]::WriteAllText($StatusLiteral, 'SUCCESS')
  exit 0
}
catch {
  Write-Error (`$_ | Out-String)
  [System.IO.File]::WriteAllText($StatusLiteral, 'FAILED')
  exit 1
}
"@
}

function Invoke-ParallelTasksetWorkers {
  param([string[]]$SeedList)
  $LogRoot = Join-Path $FormalRoot "parallel_logs"
  New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
  $Pending = New-Object System.Collections.Queue
  foreach ($Seed in $SeedList) { $Pending.Enqueue([string]$Seed) }
  $Running = @{}
  $Completed = @()
  $Failed = @()
  $PowerShellExe = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName

  while (($Pending.Count -gt 0) -or ($Running.Count -gt 0)) {
    while (($Pending.Count -gt 0) -and ($Running.Count -lt $ParallelTasksets)) {
      $Seed = [string]$Pending.Dequeue()
      $StdoutPath = Join-Path $LogRoot "seed_${Seed}.stdout.log"
      $StderrPath = Join-Path $LogRoot "seed_${Seed}.stderr.log"
      $StatusPath = Join-Path $LogRoot "seed_${Seed}.worker_status.txt"
      if (Test-Path $StdoutPath) { Remove-Item -Force $StdoutPath }
      if (Test-Path $StderrPath) { Remove-Item -Force $StderrPath }
      if (Test-Path $StatusPath) { Remove-Item -Force $StatusPath }
      $WorkerCommand = New-SeedWorkerCommand -Seed $Seed -StatusPath $StatusPath
      $EncodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($WorkerCommand))
      $Process = Start-Process -FilePath $PowerShellExe `
        -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $EncodedCommand) `
        -PassThru -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath
      $Running[$Seed] = [pscustomobject]@{
        Seed = $Seed
        Process = $Process
        StdoutPath = $StdoutPath
        StderrPath = $StderrPath
        StatusPath = $StatusPath
      }
      Write-Host "[parallel] START seed=$Seed running=$($Running.Count)/$ParallelTasksets"
    }

    if ($Running.Count -gt 0) { Start-Sleep -Seconds 2 }
    foreach ($Seed in @($Running.Keys)) {
      $Entry = $Running[$Seed]
      if (-not $Entry.Process.HasExited) { continue }

      # Wait for the process handle and redirected streams to be fully finalized before
      # reading its state. The worker also writes an explicit terminal status file just
      # before exiting, which avoids the intermittent blank ExitCode observed with
      # Start-Process -PassThru on this parallel runner.
      $Entry.Process.WaitForExit()
      $Entry.Process.Refresh()

      $WorkerStatus = ""
      if (Test-Path $Entry.StatusPath) {
        $WorkerStatus = (Get-Content -Path $Entry.StatusPath -Raw).Trim()
      }

      $ExitCode = $null
      try {
        $ExitCode = $Entry.Process.ExitCode
      }
      catch {
        $ExitCode = $null
      }

      $WorkerSucceeded = ($WorkerStatus -eq "SUCCESS")
      $WorkerFailed = ($WorkerStatus -eq "FAILED")

      if ((-not $WorkerSucceeded) -and (-not $WorkerFailed) -and ($null -ne $ExitCode)) {
        $WorkerSucceeded = ([int]$ExitCode -eq 0)
        $WorkerFailed = (-not $WorkerSucceeded)
      }

      $Running.Remove($Seed)
      if ($WorkerSucceeded) {
        $Completed += [int]$Seed
        Write-Host "[parallel] DONE seed=$Seed"
      } else {
        $Failed += [int]$Seed
        $ExitCodeText = if ($null -eq $ExitCode) { "<unavailable>" } else { [string]$ExitCode }
        $StatusText = if ([string]::IsNullOrWhiteSpace($WorkerStatus)) { "<missing>" } else { $WorkerStatus }
        $Message = "seed=$Seed worker failed; worker_status=$StatusText; exit_code=$ExitCodeText; stdout=$($Entry.StdoutPath); stderr=$($Entry.StderrPath)"
        Write-Host $Message -ForegroundColor Red
        Add-Content -Path $GlobalFailureLog -Value $Message
        if (Test-Path $Entry.StdoutPath) { Get-Content $Entry.StdoutPath -Tail 60 | ForEach-Object { Write-Host $_ } }
        if (Test-Path $Entry.StderrPath) { Get-Content $Entry.StderrPath -Tail 60 | ForEach-Object { Write-Host $_ -ForegroundColor Red } }
        if (-not $ContinueOnSeedFailure) {
          foreach ($OtherSeed in @($Running.Keys)) {
            try { Stop-Process -Id $Running[$OtherSeed].Process.Id -Force -ErrorAction SilentlyContinue } catch {}
          }
          throw $Message
        }
      }
    }
  }
  return [pscustomobject]@{ Completed = @($Completed); Failed = @($Failed) }
}

Assert-Exists -Path "scripts\evaluate_dqn_amc.py" -Message "Missing evaluator. Wrong ProjectRoot?"
Assert-Exists -Path "scripts\select_csem_static_budget_baseline.py" -Message "Missing static selector."
Assert-Exists -Path "scripts\select_csem_pressure_heuristic.py" -Message "Missing pressure selector."
Assert-Exists -Path "scripts\aggregate_csem_baseline_suite.py" -Message "Missing aggregate script."
if ($EvaluationWorkers -lt 1) { throw "EvaluationWorkers must be >= 1." }
if ($ParallelTasksets -lt 1) { throw "ParallelTasksets must be >= 1." }

New-Item -ItemType Directory -Force -Path $FormalRoot | Out-Null
if ($Force -and (-not $SeedWorker) -and (Test-Path $GlobalFailureLog)) {
  Remove-Item -Force $GlobalFailureLog
}

Write-Host "============================================================"
Write-Host "C-AMC-sem formal10 four-baseline runner"
Write-Host "FormalRoot:       $FormalRoot"
Write-Host "Taskset seeds:    $TasksetSeeds"
Write-Host "Validation:       $ValidationSeeds ; h=$ValidationEndTime"
Write-Host "Final HOUT:       $HoutSeeds ; h=$HoutEndTimes"
Write-Host "Agent period:     $AgentPeriod"
Write-Host "Parallel tasksets: $ParallelTasksets ; eval workers=$EvaluationWorkers"
Write-Host "============================================================"

$TasksetSeedList = Parse-List -Raw $TasksetSeeds
$UseParallel = (-not $SeedWorker) -and ($ParallelTasksets -gt 1) -and ($TasksetSeedList.Count -gt 1)
$CompletedSeeds = @()
$FailedSeeds = @()
if ($UseParallel) {
  $Result = Invoke-ParallelTasksetWorkers -SeedList $TasksetSeedList
  $CompletedSeeds = @($Result.Completed)
  $FailedSeeds = @($Result.Failed)
} else {
  foreach ($SeedText in $TasksetSeedList) {
    $TasksetSeed = [int]$SeedText
    try {
      Invoke-TasksetBaselineSuite -TasksetSeed $TasksetSeed
      $CompletedSeeds += $TasksetSeed
      Write-Host "DONE seed=$TasksetSeed"
    }
    catch {
      $Message = "seed=$TasksetSeed failed: $($_.Exception.Message)"
      Write-Host $Message -ForegroundColor Red
      $SeedFailureLog = if ($SeedWorker) {
        Join-Path (Join-Path $FormalRoot "r0_s$TasksetSeed") "seed_worker_failure.txt"
      } else { $GlobalFailureLog }
      $FailureParent = Split-Path -Parent $SeedFailureLog
      if ($FailureParent -and !(Test-Path $FailureParent)) {
        New-Item -ItemType Directory -Force -Path $FailureParent | Out-Null
      }
      Add-Content -Path $SeedFailureLog -Value $Message
      $FailedSeeds += $TasksetSeed
      if (-not $ContinueOnSeedFailure) { throw }
    }
  }
}

if ((-not $SeedWorker) -and (-not $SkipHout)) {
  $AggregateDir = Join-Path $FormalRoot "baseline_aggregate"
  Invoke-CondaPython -Arguments @(
    "python", "-u", "scripts\aggregate_csem_baseline_suite.py",
    "--formal-root", $FormalRoot,
    "--taskset-seeds", $TasksetSeeds,
    "--output-dir", $AggregateDir
  ) | Out-Null
}

Write-Host "============================================================"
Write-Host "DONE: C-AMC-sem formal10 four-baseline runner"
Write-Host "Completed seeds: $($CompletedSeeds -join ',')"
if ($FailedSeeds.Count -gt 0) { Write-Host "Failed seeds:    $($FailedSeeds -join ',')" -ForegroundColor Yellow }
Write-Host "Formal root:     $FormalRoot"
if (-not $SeedWorker) { Write-Host "Aggregate dir:   $(Join-Path $FormalRoot 'baseline_aggregate')" }
if ((-not $SeedWorker) -and (Test-Path $GlobalFailureLog)) { Write-Host "Failure log:     $GlobalFailureLog" }
Write-Host "============================================================"
