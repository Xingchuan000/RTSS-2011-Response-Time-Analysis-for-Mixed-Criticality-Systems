param(
  [string]$ProjectRoot = "D:\AMC",
  [string]$CondaEnv = "amc-repro-macosmatch",

  [ValidateSet("v11_full_10d", "v14_qamc_full_12d")]
  [string[]]$ObservationModes = @(
    "v11_full_10d",
    "v14_qamc_full_12d"
  ),

  [int[]]$TasksetSeeds = @(2221, 2535),
  [int[]]$TrainingRandomSeeds = @(0, 1, 2),
  [int]$Episodes = 600,
  [long]$TrainEndTime = 5000000,
  [string]$ValidationSeeds = "200:209",
  [long]$ValidationEndTime = 5000000,
  [string]$HoutSeeds = "1550:1559",
  [long[]]$HoutEndTimes = @(5000000, 50000000),
  [int]$ValidationWorkers = 12,
  [int]$EvaluationWorkers = 2,
  [switch]$Pilot,
  [switch]$Force,

  [string]$CAmcReferenceTrainRoot = "outputs\dcs_t10_e1350\tr",
  [string]$QAmcProfileSpec = "configs\qamc_profile_spec_v2.json"
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

$ValidateEvery = 10
$CheckpointEvery = 10
if ($Pilot) {
  $TasksetSeeds = @(2221)
  $TrainingRandomSeeds = @(0)
  $Episodes = 2
  $TrainEndTime = 100000
  $ValidationSeeds = "200"
  $ValidationEndTime = 100000
  $HoutSeeds = "1550"
  $HoutEndTimes = @(100000)
  $ValidateEvery = 1
  $CheckpointEvery = 1
}
if ($Episodes -lt 1) {
  throw "Episodes must be at least 1."
}

$OutputRoot = "outputs\qamc_observation_o2"
if ($Pilot) {
  $OutputRoot = "$($OutputRoot)_pilot"
}
$TrainRoot = "$OutputRoot\train"
$HoutRoot = "$OutputRoot\hout"
$ArtifactRoot = "$OutputRoot\artifacts"
$BaseReferenceRoot = "$ArtifactRoot\base_reference"
$ObservationReferenceRoot = "$ArtifactRoot\observation_reference"
$ProfileRoot = "$ArtifactRoot\profiles"
$ProfileManifest = "$ProfileRoot\manifest.json"
$RewardMode = "interval_qos_v2_single_recovery_full_C5_overinc016_abs005"

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

function Invoke-Python {
  param([string[]]$Arguments)
  & conda run --no-capture-output -n $CondaEnv python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
  }
}

function Get-ModelPath {
  param([string]$RunDir)
  foreach ($Candidate in @(
    "$RunDir\model_best_qos_recovery_stable.pt",
    "$RunDir\model_best.pt",
    "$RunDir\model_final.pt"
  )) {
    if (Test-Path $Candidate) {
      return $Candidate
    }
  }
  throw "No checkpoint found in $RunDir"
}

function Get-HorizonLabel {
  param([long]$EndTime)
  if ($EndTime -eq 5000000) { return "h5e6" }
  if ($EndTime -eq 50000000) { return "h5e7" }
  return "t$EndTime"
}

function Ensure-TasksetArtifacts {
  param([int]$TasksetSeed)

  $ReferenceRunDir = "$CAmcReferenceTrainRoot\r0_s$TasksetSeed"
  if (!(Test-Path "$ReferenceRunDir\config.json")) {
    throw "Missing O0 C-AMC-sem reference run: $ReferenceRunDir"
  }
  New-Item -ItemType Directory -Force -Path $BaseReferenceRoot | Out-Null
  New-Item -ItemType Directory -Force -Path $ProfileRoot | Out-Null
  $BaseFrozen = "$BaseReferenceRoot\r0_s$TasksetSeed.frozen.json"
  if ($Force -or !(Test-Path $BaseFrozen)) {
    Invoke-Python -Arguments @(
      "-u", "scripts\freeze_qamc_reference_config.py",
      "--reference-run-dir", $ReferenceRunDir,
      "--project-root", $ProjectRoot,
      "--allow-legacy-upgrade",
      "--output", $BaseFrozen
    )
  }
  Invoke-Python -Arguments @(
    "-u", "scripts\materialize_qamc_profiles.py",
    "--reference-run-dir", $ReferenceRunDir,
    "--spec", $QAmcProfileSpec,
    "--output-dir", $ProfileRoot,
    "--append-manifest"
  )
  return $BaseFrozen
}

function Get-ReferenceForMode {
  param(
    [int]$TasksetSeed,
    [string]$ObservationMode,
    [string]$BaseFrozen
  )
  if ($ObservationMode -eq "v11_full_10d") {
    return $BaseFrozen
  }

  $VariantDir = "$ObservationReferenceRoot\$ObservationMode\r0_s$TasksetSeed"
  $VariantFrozen = "$ObservationReferenceRoot\$ObservationMode\r0_s$TasksetSeed.frozen.json"
  if ($Force -or !(Test-Path $VariantFrozen)) {
    Invoke-Python -Arguments @(
      "-u", "scripts\derive_qamc_observation_reference.py",
      "--base-frozen-reference", $BaseFrozen,
      "--observation-mode", $ObservationMode,
      "--output-dir", $VariantDir
    )
    Invoke-Python -Arguments @(
      "-u", "scripts\freeze_qamc_reference_config.py",
      "--reference-run-dir", $VariantDir,
      "--project-root", $ProjectRoot,
      "--output", $VariantFrozen
    )
  }
  return $VariantFrozen
}

function Invoke-Train {
  param(
    [int]$TasksetSeed,
    [int]$TrainingSeed,
    [string]$ObservationMode,
    [string]$FrozenReference,
    [string]$RunDir
  )
  if ((Test-Path "$RunDir\model_final.pt") -and (-not $Force)) {
    Write-Host "SKIP completed train run: $RunDir"
    return
  }
  New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
  $TrainSeeds = "0:$($Episodes - 1)"
  $PolicyArgs = @(
    "--agent-period", "25000",
    "--reward-mode", $RewardMode,
    "--action-space", "single",
    "--observation-mode", $ObservationMode,
    "--budget-increase-ratio", "0.02",
    "--budget-decrease-ratio", "0.02",
    "--budget-floor-ratio", "0.9",
    "--forbid-decreasing-hi-budgets",
    "--mask-detail-mode", "full",
    "--enable-deploy-cap-mask",
    "--deploy-cap-mask-ratio", "4.0",
    "--deploy-cap-mask-criticality", "lo"
  )
  $Arguments = @("-u", "scripts\train_dqn_amc.py") + $WorkloadArgs + @(
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
    "--dqn-runtime-semantics", "Q_AMC",
    "--validation-baseline-semantics", "Q_AMC"
  ) + $PolicyArgs + @(
    "--save-best-by", "qos_recovery_stable",
    "--require-better-than-baseline-for-best",
    "--save-all-best-types",
    "--double-dqn",
    "--hidden-layers", "128,128",
    "--learning-rate", "5e-5",
    "--batch-size", "64",
    "--replay-capacity", "10000",
    "--min-replay-size", "500",
    "--target-update-freq", "5",
    "--gamma", "0.99",
    "--epsilon-start", "1.0",
    "--epsilon-end", "0.05",
    "--epsilon-decay-steps", "8000",
    "--log-validation-policy-actions",
    "--dqn-device", "cuda",
    "--seed", "$TrainingSeed",
    "--network-seed", "$TrainingSeed",
    "--exploration-seed", "$TrainingSeed",
    "--replay-seed", "$TrainingSeed",
    "--qamc-reference-config-path", $FrozenReference,
    "--qamc-profile-manifest-path", $ProfileManifest,
    "--qamc-profile-spec-path", $QAmcProfileSpec,
    "--output-dir", $RunDir
  )
  Invoke-Python -Arguments $Arguments
}

function Invoke-Hout {
  param(
    [int]$TasksetSeed,
    [int]$TrainingSeed,
    [string]$ObservationMode,
    [string]$FrozenReference,
    [string]$RunDir,
    [long]$EndTime
  )
  $Label = Get-HorizonLabel -EndTime $EndTime
  $OutputCsv = "$HoutRoot\$ObservationMode\r$($TrainingSeed)_s$TasksetSeed\$Label.csv"
  if ((Test-Path $OutputCsv) -and (-not $Force)) {
    Write-Host "SKIP completed evaluation: $OutputCsv"
    return
  }
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputCsv) | Out-Null
  $ModelPath = Get-ModelPath -RunDir $RunDir
  $Arguments = @("-u", "scripts\evaluate_dqn_amc.py") + $WorkloadArgs + @(
    "--fixed-taskset-seed", "$TasksetSeed",
    "--scenario-seed-offset", "100000",
    "--model", $ModelPath,
    "--seeds", $HoutSeeds,
    "--evaluation-workers", "$EvaluationWorkers",
    "--end-time", "$EndTime",
    "--scenario", "stress",
    "--dqn-runtime-semantics", "Q_AMC",
    "--baselines", "q_amc_native,q_amc_dqn_budget_overlay",
    "--agent-period", "25000",
    "--reward-mode", $RewardMode,
    "--action-space", "single",
    "--observation-mode", $ObservationMode,
    "--budget-increase-ratio", "0.02",
    "--budget-decrease-ratio", "0.02",
    "--budget-floor-ratio", "0.9",
    "--forbid-decreasing-hi-budgets",
    "--mask-detail-mode", "full",
    "--enable-deploy-cap-mask",
    "--deploy-cap-mask-ratio", "4.0",
    "--deploy-cap-mask-criticality", "lo",
    "--qamc-reference-config-path", $FrozenReference,
    "--qamc-profile-manifest-path", $ProfileManifest,
    "--qamc-profile-spec-path", $QAmcProfileSpec,
    "--output", $OutputCsv
  )
  Invoke-Python -Arguments $Arguments
}

foreach ($RequiredPath in @(
  "scripts\train_dqn_amc.py",
  "scripts\evaluate_dqn_amc.py",
  "scripts\derive_qamc_observation_reference.py",
  "scripts\freeze_qamc_reference_config.py",
  "scripts\materialize_qamc_profiles.py",
  $QAmcProfileSpec
)) {
  if (!(Test-Path $RequiredPath)) {
    throw "Missing required file: $RequiredPath"
  }
}

foreach ($TasksetSeed in $TasksetSeeds) {
  $BaseFrozen = Ensure-TasksetArtifacts -TasksetSeed $TasksetSeed
  foreach ($ObservationMode in $ObservationModes) {
    $FrozenReference = Get-ReferenceForMode `
      -TasksetSeed $TasksetSeed `
      -ObservationMode $ObservationMode `
      -BaseFrozen $BaseFrozen
    foreach ($TrainingSeed in $TrainingRandomSeeds) {
      $RunDir = "$TrainRoot\$ObservationMode\r$($TrainingSeed)_s$TasksetSeed"
      Invoke-Train `
        -TasksetSeed $TasksetSeed `
        -TrainingSeed $TrainingSeed `
        -ObservationMode $ObservationMode `
        -FrozenReference $FrozenReference `
        -RunDir $RunDir
      foreach ($EndTime in $HoutEndTimes) {
        Invoke-Hout `
          -TasksetSeed $TasksetSeed `
          -TrainingSeed $TrainingSeed `
          -ObservationMode $ObservationMode `
          -FrozenReference $FrozenReference `
          -RunDir $RunDir `
          -EndTime $EndTime
      }
    }
  }
}

Write-Host "QAMC_OBSERVATION_O2_MATRIX_COMPLETED"
