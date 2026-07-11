# fixed-ranked VIPER 最小 smoke 入口；输出目录与历史 top1 实验完全隔离。
# 训练结束后自动执行 HOUT 评估，验证 fixed-point ranked 语义。
param(
    [Parameter(Mandatory = $true)][string]$TeacherModel,
    [string]$OutputRoot = "outputs\viper_fixed_ranked_v1_smoke"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
$Dataset = Join-Path $OutputRoot "dataset"
$Trees = Join-Path $OutputRoot "trees"

# Step 1: 采集 teacher-only dataset
conda run -n amc-repro python scripts/collect_viper_teacher_data.py `
    --model $TeacherModel --teacher-id fixed_ranked_smoke --workload small `
    --action-space single --tree-state-encoding fixed_point_int `
    --tree-fallback-mode ranked_valid_or_none --fixed-ranked-deployment-v1 `
    --action-validation-mode formal_v1 --strict-candidate-deploy-cap `
    --carry-over-aware-safety --lo-budget-overrun-guard-units 1 `
    --require-integer-tree-artifact --output-dir $Dataset

# Step 2: 训练 VIPER 树
conda run -n amc-repro python scripts/train_viper_tree.py `
    --method viper --teacher-model $TeacherModel --teacher-id fixed_ranked_smoke `
    --initial-dataset $Dataset --workload small --action-space single `
    --fixed-ranked-deployment-v1 --tree-state-encoding fixed_point_int `
    --tree-fallback-mode ranked_valid_or_none --tree-selection-mode performance_compatible `
    --action-validation-mode formal_v1 --strict-candidate-deploy-cap `
    --carry-over-aware-safety --lo-budget-overrun-guard-units 1 `
    --require-integer-tree-artifact --output-dir $Trees

# Step 3: 定位 best/ artifact 目录（train_viper_tree.py 在 depth_2/leaf_1/best 下）
$BestDir = Get-ChildItem -Path $Trees -Recurse -Directory -Filter "best" | Select-Object -First 1
if (-not $BestDir) {
    throw "未找到 best/ artifact 目录"
}
# Step 4: 定位 teacher model 路径
$TeacherModelPath = $TeacherModel
if (-not (Test-Path $TeacherModelPath)) {
    $TeacherModelPath = Join-Path (Join-Path $Trees "..") "teacher" "model_final.pt"
}

# Step 5: 执行 fixed-ranked HOUT 评估
$HoutCsv = Join-Path $OutputRoot "evaluate_dqn_amc_fixed_ranked_smoke_hout.csv"
conda run -n amc-repro python scripts/evaluate_dqn_amc.py `
    --model $TeacherModelPath `
    --bc-tree-model $BestDir.FullName `
    --tree-compare-teacher-model $TeacherModelPath `
    --baselines "bc_tree_agent" `
    --workload small `
    --seeds "0" `
    --end-time 20000 `
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
    --output $HoutCsv `
    --evaluation-workers 1

# Step 6: 验证 HOUT CSV 存在且包含 tree 方法行
if (-not (Test-Path $HoutCsv)) {
    throw "HOUT CSV 未生成: $HoutCsv"
}
$lines = Get-Content $HoutCsv
if ($lines.Count -lt 2) {
    throw "HOUT CSV 没有数据行"
}
# 不只确认文件生成：正式 HOUT 必须产出 tree 行并保留完整 fixed-ranked 语义。
$TreeRow = Import-Csv $HoutCsv | Where-Object { $_.method -eq "bc_tree_agent" } | Select-Object -First 1
if (-not $TreeRow) { throw "HOUT CSV 缺少 bc_tree_agent 数据行" }
if ($TreeRow.semantic_validation_passed -ne "True") { throw "HOUT 部署语义校验未通过" }
if ($TreeRow.artifact_schema_version -ne "viper_integer_ranked_artifact_v2") { throw "HOUT artifact schema 不正确" }
if ($TreeRow.tree_fallback_mode -ne "ranked_valid_or_none") { throw "HOUT ranked fallback 语义不正确" }
if ([int]$TreeRow.formal_v1_mask_step_mismatch_count -ne 0) { throw "HOUT formal_v1 mask step mismatch 非零" }
Write-Host "Fixed-ranked smoke 完成，HOUT: $HoutCsv"
