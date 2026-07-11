# fixed-ranked VIPER 正式搜索模板。该脚本使用独立后缀，不覆盖旧 top1/noop 结果。
# 训练结束后分别在 2e7 和 5e7 时域执行 HOUT 评估。
param(
    [Parameter(Mandatory = $true)][string]$TeacherModel,
    [Parameter(Mandatory = $true)][string]$InitialDataset,
    [string]$OutputRoot = "outputs\viper_fixed_ranked_v1_h2_h5",
    # 直接透传 --seeds，支持单 seed、范围（如 0:4）和逗号列表。
    [string]$Seeds = "0",
    [string]$EvaluationWorkers = "4"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Step 1: 训练树（depth 2-5 scan）
conda run -n amc-repro python scripts/train_viper_tree.py `
    --method viper --teacher-model $TeacherModel --teacher-id fixed_ranked_h2_h5 `
    --initial-dataset $InitialDataset --workload small --action-space single `
    --max-depth-grid "2,3,4,5" --fixed-ranked-deployment-v1 `
    --tree-state-encoding fixed_point_int --tree-fallback-mode ranked_valid_or_none `
    --tree-selection-mode performance_compatible --action-validation-mode formal_v1 `
    --strict-candidate-deploy-cap --carry-over-aware-safety `
    --lo-budget-overrun-guard-units 1 --require-integer-tree-artifact `
    --output-dir $OutputRoot

# Step 2: 定位 best/ artifact 目录
$BestDir = Get-ChildItem -Path $OutputRoot -Recurse -Directory -Filter "best" | Select-Object -First 1
if (-not $BestDir) {
    throw "未找到 best/ artifact 目录"
}

# Step 3: HOUT at 2e7 (h2)
$HoutH2Dir = Join-Path $OutputRoot "hout_h2"
New-Item -ItemType Directory -Force -Path $HoutH2Dir | Out-Null
$HoutH2Csv = Join-Path $HoutH2Dir "eval_summary.csv"
conda run -n amc-repro python scripts/evaluate_dqn_amc.py `
    --model $TeacherModel `
    --bc-tree-model $BestDir.FullName `
    --tree-compare-teacher-model $TeacherModel `
    --baselines "bc_tree_agent" `
    --workload small `
    --seeds $Seeds `
    --end-time 20000000 `
    --agent-period 20 `
    --observation-mode v11_full_10d `
    --action-space single `
    --fixed-ranked-deployment-v1 `
    --require-integer-tree-artifact `
    --tree-state-encoding fixed_point_int `
    --tree-fallback-mode ranked_valid_or_none `
    --action-validation-mode formal_v1 `
    --strict-candidate-deploy-cap `
    --carry-over-aware-safety `
    --lo-budget-overrun-guard-units 1 `
    --output $HoutH2Csv `
    --evaluation-workers $EvaluationWorkers

# 每个时域都单独验收语义，避免仅凭 CSV 存在就把错误部署结果当作 HOUT。
function Assert-FixedRankedHoutCsv([string]$CsvPath) {
    if (-not (Test-Path $CsvPath)) { throw "HOUT CSV 未生成: $CsvPath" }
    $TreeRow = Import-Csv $CsvPath | Where-Object { $_.method -eq "bc_tree_agent" } | Select-Object -First 1
    if (-not $TreeRow) { throw "HOUT CSV 缺少 bc_tree_agent 数据行: $CsvPath" }
    if ($TreeRow.semantic_validation_passed -ne "True") { throw "HOUT 部署语义校验未通过: $CsvPath" }
    if ($TreeRow.artifact_schema_version -ne "viper_integer_ranked_artifact_v2") { throw "artifact schema 不正确: $CsvPath" }
    if ($TreeRow.tree_fallback_mode -ne "ranked_valid_or_none") { throw "ranked fallback 语义不正确: $CsvPath" }
    if ([int]$TreeRow.formal_v1_mask_step_mismatch_count -ne 0) { throw "formal_v1 mask step mismatch 非零: $CsvPath" }
}
Assert-FixedRankedHoutCsv $HoutH2Csv

# Step 4: HOUT at 5e7 (h5)
$HoutH5Dir = Join-Path $OutputRoot "hout_h5"
New-Item -ItemType Directory -Force -Path $HoutH5Dir | Out-Null
$HoutH5Csv = Join-Path $HoutH5Dir "eval_summary.csv"
conda run -n amc-repro python scripts/evaluate_dqn_amc.py `
    --model $TeacherModel `
    --bc-tree-model $BestDir.FullName `
    --tree-compare-teacher-model $TeacherModel `
    --baselines "bc_tree_agent" `
    --workload small `
    --seeds $Seeds `
    --end-time 50000000 `
    --agent-period 20 `
    --observation-mode v11_full_10d `
    --action-space single `
    --fixed-ranked-deployment-v1 `
    --require-integer-tree-artifact `
    --tree-state-encoding fixed_point_int `
    --tree-fallback-mode ranked_valid_or_none `
    --action-validation-mode formal_v1 `
    --strict-candidate-deploy-cap `
    --carry-over-aware-safety `
    --lo-budget-overrun-guard-units 1 `
    --output $HoutH5Csv `
    --evaluation-workers $EvaluationWorkers

Assert-FixedRankedHoutCsv $HoutH5Csv

Write-Host "Fixed-ranked h2/h5 完成。"
Write-Host "  h2 HOUT: $HoutH2Csv"
Write-Host "  h5 HOUT: $HoutH5Csv"
