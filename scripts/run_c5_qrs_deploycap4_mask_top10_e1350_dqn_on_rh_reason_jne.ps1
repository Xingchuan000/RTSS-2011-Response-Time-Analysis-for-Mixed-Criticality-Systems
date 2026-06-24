# DQN-on-RH 与 reason-level JNE 评估入口脚本。
# 本脚本只封装计划文档中要求的训练 / HOUT 命令与最小 preflight 检查，
# 不额外加入计划边界之外的容错或自动修复逻辑。

param(
    [switch]$TrainOnly,
    [switch]$EvalOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-FileContains {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Pattern
    )

    if (-not (Test-Path $Path)) {
        throw "Required file not found: $Path"
    }
    $content = Get-Content -Path $Path -Raw -Encoding UTF8
    if (-not $content.Contains($Pattern)) {
        throw "Pattern '$Pattern' not found in $Path"
    }
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Assert-FileContains "scripts\train_dqn_amc.py" "dqn-runtime-semantics"
Assert-FileContains "scripts\evaluate_dqn_amc.py" "dqn-runtime-semantics"
Assert-FileContains "scripts\evaluate_dqn_amc.py" "lo_active_dropped_on_mode_switch"
Assert-FileContains "amc_py\runtime_models.py" "LoJobLossEvent"

$TrainRoot = "outputs\dqn_on_rh\c5_qrs_deploycap4_mask_top10_e1350"
$HoutRoot = "outputs\hout_dqn_on_rh_reason_jne"
$ModelPath = Join-Path $TrainRoot "model_final.pt"
$EvalOutput = Join-Path $HoutRoot "eval_summary.csv"

if (-not $EvalOnly) {
    conda run -n amc-repro python scripts/train_dqn_amc.py `
        --episodes 1350 `
        --end-time 1000 `
        --dqn-runtime-semantics AMC_RH `
        --validation-baseline-semantics AMC_RH `
        --enable-deploy-cap-mask `
        --deploy-cap-mask-ratio 4.0 `
        --output-dir $TrainRoot
}

if (-not $TrainOnly) {
    conda run -n amc-repro python scripts/evaluate_dqn_amc.py `
        --model $ModelPath `
        --seeds 0:9 `
        --end-time 1000 `
        --dqn-runtime-semantics AMC_RH `
        --baselines "amc_plus_baseline,amc_ra_baseline,amc_rh_baseline,noop_agent,dqn_agent" `
        --output $EvalOutput
}
