param(
  [string]$ProjectRoot = ".",
  [string]$Python = "python",
  [string]$TeacherModel = "outputs\dcs_t10_stratdyn_v1_e1350\tr\r0_s0\model_best.pt",
  [string]$OutputRoot = "outputs\dcs_t10_stratdyn_v1_e1350\viper\stratdyn_v1",
  [string]$Seeds = "0:9",
  [int]$EndTime = 100000,
  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:PYTHONPATH = "."

$Common = @(
  "--workload", "mc_stratified_dynamic",
  "--scenario", "stress",
  "--mc-strat-dyn-num-tasks", "12",
  "--mc-strat-dyn-hi-ratio", "0.5",
  "--mc-strat-dyn-period-family", "seed_paired",
  "--mc-strat-dyn-period-scale", "500",
  "--require-schedulable",
  "--dqn-runtime-semantics", "C_AMC_SEM",
  "--c-amc-sem-xf", "0.5",
  "--agent-period", "25000",
  "--action-space", "single",
  "--observation-mode", "v11_full_10d"
)

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$Dataset = Join-Path $OutputRoot "teacher_dataset"
$TrainTree = Join-Path $OutputRoot "tree"
$CollectArgs = @(
  "scripts/collect_viper_teacher_data.py",
  "--model", $TeacherModel,
  "--teacher-id", "mc_stratified_dynamic_primary10_teacher",
  "--output-dir", $Dataset,
  "--seeds", $Seeds,
  "--end-time", $EndTime
) + $Common

Write-Host "collect VIPER teacher data: $Python $($CollectArgs -join ' ')"
if (-not $DryRun) {
  & $Python @CollectArgs
  if ($LASTEXITCODE -ne 0) { throw "collect VIPER teacher data failed with exit code $LASTEXITCODE" }
}

$TrainArgs = @(
  "scripts/train_viper_tree.py",
  "--method", "viper",
  "--teacher-model", $TeacherModel,
  "--teacher-id", "mc_stratified_dynamic_primary10_teacher",
  "--initial-dataset", $Dataset,
  "--output-dir", $TrainTree,
  "--train-seeds", $Seeds,
  "--validation-seeds", $Seeds,
  "--iterations", "1"
) + $Common
Write-Host "train VIPER tree: $Python $($TrainArgs -join ' ')"
if (-not $DryRun) {
  & $Python @TrainArgs
  if ($LASTEXITCODE -ne 0) { throw "train VIPER tree failed with exit code $LASTEXITCODE" }
}
