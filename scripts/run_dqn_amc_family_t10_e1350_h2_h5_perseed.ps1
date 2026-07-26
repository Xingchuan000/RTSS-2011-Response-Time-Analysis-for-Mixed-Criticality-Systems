param(
  [string]$ProjectRoot = "D:\AMC",
  [string]$CondaEnv = "amc-repro-macosmatch",

  [ValidateSet("C_AMC_SEM", "Q_AMC")]
  [string]$RuntimeSemantics = "C_AMC_SEM",

  [string]$CAmcReferenceTrainRoot = "outputs\dcs_t10_e1350\tr",
  [string]$QAmcProfileSpec = "configs\qamc_profile_spec_v2.json",

  [int]$ValidationWorkers = 10,
  [int]$EvaluationWorkers = 1,
  [switch]$Force,
  [switch]$SkipTrain,
  [switch]$SkipHout,
  [switch]$PrepareArtifactsOnly,
  [switch]$Pilot,
  [int[]]$TasksetSeeds = @(
    2221, 397, 861, 639, 1264,
    1502, 358, 185, 2535, 2829
  ),
  [int[]]$PilotTasksetSeeds = @(185),
  [int]$Episodes = 1350,
  [long]$TrainEndTime = 5000000,
  [string]$ValidationSeeds = "200:209",
  [long]$ValidationEndTime = 5000000,
  [string]$HoutSeeds = "200:249",
  [long]$HoutH2EndTime = 20000000,
  [long]$HoutH5EndTime = 50000000
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

$RewardMode = "interval_qos_v2_single_recovery_full_C5_overinc016_abs005"
$CAmcSemXf = 0.5
$ValidateEvery = 10
$CheckpointEvery = 10
$LearningRateSchedule = "0:5e-5,450:2.5e-5,900:1.25e-5"

if ($Pilot) {
  $TasksetSeeds = $PilotTasksetSeeds
  $Episodes = 2
  $TrainEndTime = 100000
  $ValidationSeeds = "200"
  $ValidationEndTime = 100000
  $HoutSeeds = "200:201"
  $HoutH2EndTime = 100000
  $HoutH5EndTime = 100000
  $ValidateEvery = 1
  $CheckpointEvery = 1
  $LearningRateSchedule = "0:5e-5"
}

if ($Episodes -lt 1) {
  throw "Episodes must be at least 1."
}
$TrainSeeds = "0:$($Episodes - 1)"

switch ($RuntimeSemantics) {
  "C_AMC_SEM" {
    $SemanticsConfig = [ordered]@{
      DqnRuntimeSemantics = "C_AMC_SEM"
      ValidationBaselineSemantics = "C_AMC_SEM"
      OutRoot = "outputs\dcs_t10_e1350"
      HoutSubdir = "allbase_xf05_qm"
      BaselineMethods = @(
        "amc_plus_baseline",
        "amc_ra_baseline",
        "amc_rh_baseline",
        "c_amc_sem_baseline",
        "noop_agent",
        "dqn_agent"
      )
      RequiredMetricColumns = @(
        "lo_equiv_jne",
        "lo_equiv_jne_rate",
        "lo_quality_qos",
        "lo_quality_loss",
        "lo_degraded_completion_ratio",
        "lo_zero_service_ratio"
      )
    }
  }
  "Q_AMC" {
    $SemanticsConfig = [ordered]@{
      DqnRuntimeSemantics = "Q_AMC"
      ValidationBaselineSemantics = "Q_AMC"
      OutRoot = "outputs\dqs_t10_e1350"
      HoutSubdir = "allbase_qamc_v5"
      BaselineMethods = @(
        "amc_same_full_sample_native",
        "q_amc_native",
        "q_amc_budget_heuristic",
        "q_amc_dqn_budget_overlay",
        "amc_plus_baseline",
        "amc_ra_baseline",
        "amc_rh_baseline",
        "c_amc_sem_baseline",
        "noop_agent"
      )
      RequiredMetricColumns = @(
        "qamc_release_count",
        "qamc_managed_release_count",
        "qamc_paper_quality_sum",
        "qamc_paper_quality_per_release",
        "qamc_normalized_quality_qos",
        "qamc_zero_service_ratio",
        "qamc_overrun_stop_count",
        "qamc_quality_transition_count",
        "qamc_min_quality_exhaustion_count",
        "qamc_trigger_budget_mean_ratio_to_c_lo",
        "qamc_profile_fingerprint",
        "qamc_legacy_degraded_metrics_applicable"
      )
    }
  }
}

$DqnRuntimeSemantics = $SemanticsConfig.DqnRuntimeSemantics
$ValidationBaselineSemantics = $SemanticsConfig.ValidationBaselineSemantics
$BaselineMethods = $SemanticsConfig.BaselineMethods -join ","
$RequiredNewMetricColumns = $SemanticsConfig.RequiredMetricColumns
$OutRoot = $SemanticsConfig.OutRoot
if ($Pilot) {
  $OutRoot = "$($OutRoot)_pilot"
}
$TrainRoot = "$OutRoot\tr"
$HoutRoot = "$OutRoot\ho\$($SemanticsConfig.HoutSubdir)"
$ArtifactRoot = "$OutRoot\artifacts"
$FrozenReferenceRoot = "$ArtifactRoot\reference"
$QAmcProfileRoot = "$ArtifactRoot\profiles"
$QAmcManifestPath = "$QAmcProfileRoot\manifest.json"

$WorkloadArgs = @(
  "--workload", "mc_fairgen",
  "--mc-fairgen-mode", "paper_learnable_headroom",
  "--mc-fairgen-num-tasks", "12",
  "--mc-fairgen-hi-ratio", "0.5",
  "--mc-fairgen-period-source", "controlled_medium",
  "--mc-fairgen-period-scale", "500",
  "--mc-fairgen-u-hi-lo-min", "0.20",
  "--mc-fairgen-u-hi-lo-max", "0.35",
  "--mc-fairgen-u-hi-hi-min", "0.45",
  "--mc-fairgen-u-hi-hi-max", "0.70",
  "--mc-fairgen-u-lo-lo-min", "0.25",
  "--mc-fairgen-u-lo-lo-max", "0.45",
  "--mc-fairgen-hi-budget-rho-min", "0.55",
  "--mc-fairgen-hi-budget-rho-max", "0.75",
  "--mc-fairgen-lo-budget-rho-min", "0.20",
  "--mc-fairgen-lo-budget-rho-max", "0.40",
  "--mc-fairgen-hi-overrun-prob", "0.08",
  "--mc-fairgen-lo-overrun-prob", "0.12",
  "--mc-fairgen-hi-overrun-factor-min", "1.02",
  "--mc-fairgen-hi-overrun-factor-max", "1.25",
  "--mc-fairgen-lo-overrun-factor-min", "1.02",
  "--mc-fairgen-lo-overrun-factor-max", "1.25"
)

$PolicyArgs = @(
  "--agent-period", "25000",
  "--c-amc-sem-xf", "$CAmcSemXf",
  "--reward-mode", $RewardMode,
  "--action-space", "single",
  "--observation-mode", "v11_full_10d",
  "--budget-increase-ratio", "0.02",
  "--budget-decrease-ratio", "0.02",
  "--budget-floor-ratio", "0.9",
  "--forbid-decreasing-hi-budgets",
  "--mask-detail-mode", "full",
  "--enable-deploy-cap-mask",
  "--deploy-cap-mask-ratio", "4.0",
  "--deploy-cap-mask-criticality", "lo"
)

function Invoke-PythonModule {
  param([string[]]$Arguments)
  & conda run --no-capture-output -n $CondaEnv python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE: $($Arguments -join ' ')"
  }
}

function Get-EndTimeLabel {
  param([long]$EndTime)
  if ($EndTime -eq 20000000) { return "h2" }
  if ($EndTime -eq 50000000) { return "h5" }
  return "t$EndTime"
}

function Get-ModelPath {
  param([string]$RunDir)
  foreach ($Candidate in @(
    "$RunDir\model_best_qos_recovery_stable.pt",
    "$RunDir\model_best.pt",
    "$RunDir\model_final.pt"
  )) {
    if (Test-Path $Candidate) { return $Candidate }
  }
  return $null
}

function Assert-CsvHasColumns {
  param([string]$CsvPath, [string[]]$Columns)
  if (!(Test-Path $CsvPath)) {
    throw "Expected output CSV does not exist: $CsvPath"
  }
  $HeaderColumns = (Get-Content -Path $CsvPath -TotalCount 1).Split(",")
  $Missing = @($Columns | Where-Object { $HeaderColumns -notcontains $_ })
  if ($Missing.Count -gt 0) {
    throw "Output CSV is missing required metric columns: $($Missing -join ', '); file=$CsvPath"
  }
}

function Ensure-QAmcArtifacts {
  param([int]$TasksetSeed)
  if ($RuntimeSemantics -ne "Q_AMC") { return $null }

  $ReferenceRunDir = "$CAmcReferenceTrainRoot\r0_s$TasksetSeed"
  $ReferenceConfig = "$ReferenceRunDir\config.json"
  if (!(Test-Path $ReferenceConfig)) {
    throw "Missing C-AMC-sem reference config for seed $TasksetSeed: $ReferenceConfig"
  }
  New-Item -ItemType Directory -Force -Path $FrozenReferenceRoot | Out-Null
  New-Item -ItemType Directory -Force -Path $QAmcProfileRoot | Out-Null
  $FrozenReference = "$FrozenReferenceRoot\r0_s$TasksetSeed.frozen.json"

  if ($Force -or !(Test-Path $FrozenReference)) {
    Invoke-PythonModule -Arguments @(
      "-u", "scripts\freeze_qamc_reference_config.py",
      "--reference-run-dir", $ReferenceRunDir,
      "--project-root", $ProjectRoot,
      "--allow-legacy-upgrade",
      "--output", $FrozenReference
    )
  }
  Invoke-PythonModule -Arguments @(
    "-u", "scripts\materialize_qamc_profiles.py",
    "--reference-run-dir", $ReferenceRunDir,
    "--spec", $QAmcProfileSpec,
    "--output-dir", $QAmcProfileRoot,
    "--append-manifest"
  )
  if (!(Test-Path $QAmcManifestPath)) {
    throw "q-AMC manifest was not generated: $QAmcManifestPath"
  }
  return [ordered]@{
    FrozenReference = $FrozenReference
    Manifest = $QAmcManifestPath
    Spec = $QAmcProfileSpec
  }
}

function Invoke-TrainOneRun {
  param([int]$TasksetSeed, [string]$OutDir, [object]$QAmcArtifacts)
  if ((Test-Path "$OutDir\model_final.pt") -and (-not $Force)) {
    Write-Host "SKIP existing finished train run: $OutDir"
    return
  }
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
  $TrainArgs = @("-u", "scripts\train_dqn_amc.py") + $WorkloadArgs + @(
    "--fixed-taskset-seed", "$TasksetSeed",
    "--train-seed-mode", "per-episode",
    "--train-seeds", $TrainSeeds,
    "--scenario-seed-offset", "100000",
    "--episodes", "$Episodes",
    "--end-time", "$TrainEndTime",
    "--validation-seeds", $ValidationSeeds,
    "--validation-end-time", "$ValidationEndTime",
    "--validate-every", "$ValidateEvery",
    "--checkpoint", "$CheckpointEvery",
    "--validation-workers", "$ValidationWorkers",
    "--dqn-runtime-semantics", $DqnRuntimeSemantics,
    "--validation-baseline-semantics", $ValidationBaselineSemantics
  ) + $PolicyArgs + @(
    "--save-best-by", "qos_recovery_stable",
    "--qos-stable-mode-delta", "0.05",
    "--qos-recovery-max-increase-rate", "0.90",
    "--qos-recovery-min-recovery-decrease-rate", "0.03",
    "--qos-recovery-max-over-increase-rate", "0.90",
    "--require-better-than-baseline-for-best",
    "--save-all-best-types",
    "--double-dqn",
    "--hidden-layers", "128,128",
    "--learning-rate", "5e-5",
    "--learning-rate-schedule", $LearningRateSchedule,
    "--batch-size", "64",
    "--replay-capacity", "10000",
    "--min-replay-size", "500",
    "--target-update-freq", "5",
    "--gamma", "0.99",
    "--epsilon-start", "1.0",
    "--epsilon-end", "0.05",
    "--epsilon-decay-steps", "8000",
    "--noop-exploration-prob", "0.0",
    "--log-train-metrics",
    "--log-step-every", "1",
    "--max-q-diagnostic-samples", "1000",
    "--log-validation-policy-actions",
    "--dqn-device", "cuda",
    "--seed", "0",
    "--network-seed", "0",
    "--exploration-seed", "0",
    "--replay-seed", "0",
    "--use-elite-replay",
    "--elite-replay-capacity", "2000",
    "--elite-replay-min-size", "128",
    "--elite-batch-size", "4",
    "--elite-score-min", "0.05",
    "--elite-score-ratio", "0.85",
    "--elite-recent-episodes", "10",
    "--elite-start-episode", "100",
    "--elite-max-mode-delta", "0.05",
    "--elite-require-no-hi-miss",
    "--elite-require-qos-stable",
    "--elite-max-add-per-validation", "2000",
    "--output-dir", $OutDir
  )
  if ($RuntimeSemantics -eq "Q_AMC") {
    $TrainArgs += @(
      "--qamc-reference-config-path", $QAmcArtifacts.FrozenReference,
      "--qamc-profile-manifest-path", $QAmcArtifacts.Manifest,
      "--qamc-profile-spec-path", $QAmcArtifacts.Spec
    )
  }
  Invoke-PythonModule -Arguments $TrainArgs
}

function Invoke-HoutOneRun {
  param(
    [int]$TasksetSeed,
    [string]$RunDir,
    [long]$EndTime,
    [string]$HorizonLabel,
    [object]$QAmcArtifacts
  )
  $Label = if ($HorizonLabel) {
    $HorizonLabel
  } else {
    Get-EndTimeLabel -EndTime $EndTime
  }
  $OutputCsv = "$HoutRoot\$Label\r0_s$TasksetSeed\hout_$Label.csv"
  $ModelPath = Get-ModelPath -RunDir $RunDir
  if ($null -eq $ModelPath) {
    throw "No checkpoint found for $RunDir"
  }
  if ((Test-Path $OutputCsv) -and (-not $Force)) {
    try {
      Assert-CsvHasColumns -CsvPath $OutputCsv -Columns $RequiredNewMetricColumns
      Write-Host "SKIP existing complete HOUT: $OutputCsv"
      return
    } catch {
      Write-Host "Regenerating incomplete HOUT: $OutputCsv"
    }
  }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputCsv) | Out-Null
  $EvalArgs = @("-u", "scripts\evaluate_dqn_amc.py") + $WorkloadArgs + @(
    "--fixed-taskset-seed", "$TasksetSeed",
    "--scenario-seed-offset", "100000",
    "--model", $ModelPath,
    "--seeds", $HoutSeeds,
    "--evaluation-workers", "$EvaluationWorkers",
    "--end-time", "$EndTime",
    "--scenario", "stress",
    "--dqn-runtime-semantics", $DqnRuntimeSemantics,
    "--baselines", $BaselineMethods
  ) + $PolicyArgs + @(
    "--double-dqn",
    "--max-q-diagnostic-samples", "1000",
    "--output", $OutputCsv
  )
  if ($RuntimeSemantics -eq "Q_AMC") {
    $EvalArgs += @(
      "--qamc-reference-config-path", $QAmcArtifacts.FrozenReference,
      "--qamc-profile-manifest-path", $QAmcArtifacts.Manifest,
      "--qamc-profile-spec-path", $QAmcArtifacts.Spec
    )
  }
  Invoke-PythonModule -Arguments $EvalArgs
  Assert-CsvHasColumns -CsvPath $OutputCsv -Columns $RequiredNewMetricColumns
}

foreach ($Path in @(
  "scripts\train_dqn_amc.py",
  "scripts\evaluate_dqn_amc.py",
  "configs\reward_modes\$RewardMode.json"
)) {
  if (!(Test-Path $Path)) { throw "Missing required file: $Path" }
}
if ($RuntimeSemantics -eq "Q_AMC") {
  foreach ($Path in @(
    "scripts\freeze_qamc_reference_config.py",
    "scripts\materialize_qamc_profiles.py",
    "amc_py\qamc\metrics_support.py",
    "amc_py\qamc\reference_config.py",
    $QAmcProfileSpec
  )) {
    if (!(Test-Path $Path)) { throw "Missing q-AMC required file: $Path" }
  }
}

New-Item -ItemType Directory -Force -Path $TrainRoot | Out-Null
New-Item -ItemType Directory -Force -Path $HoutRoot | Out-Null

$QAmcArtifactsBySeed = @{}
if ($RuntimeSemantics -eq "Q_AMC") {
  foreach ($TasksetSeed in $TasksetSeeds) {
    $QAmcArtifactsBySeed[$TasksetSeed] = Ensure-QAmcArtifacts -TasksetSeed $TasksetSeed
    Write-Host "Prepared artifacts for r0_s$TasksetSeed"
  }
  if ($PrepareArtifactsOnly) {
    Write-Host "ARTIFACT_PREPARATION_COMPLETED"
    exit 0
  }
}

foreach ($TasksetSeed in $TasksetSeeds) {
  $RunDir = "$TrainRoot\r0_s$TasksetSeed"
  $QAmcArtifacts = if ($RuntimeSemantics -eq "Q_AMC") {
    $QAmcArtifactsBySeed[$TasksetSeed]
  } else {
    $null
  }
  if (-not $SkipTrain) {
    Invoke-TrainOneRun -TasksetSeed $TasksetSeed -OutDir $RunDir -QAmcArtifacts $QAmcArtifacts
  }
  if (-not $SkipHout) {
    Invoke-HoutOneRun -TasksetSeed $TasksetSeed -RunDir $RunDir -EndTime $HoutH2EndTime -HorizonLabel "h2" -QAmcArtifacts $QAmcArtifacts
    Invoke-HoutOneRun -TasksetSeed $TasksetSeed -RunDir $RunDir -EndTime $HoutH5EndTime -HorizonLabel "h5" -QAmcArtifacts $QAmcArtifacts
  }
}
