"""fixed-ranked PowerShell 入口的静态回归测试。"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> str:
    """读取脚本文本；该测试只约束计划要求的正式运行参数。"""
    return (PROJECT_ROOT / "scripts" / name).read_text(encoding="utf-8")


REQUIRED_PROFILE_FLAGS = (
    "--fixed-ranked-deployment-v1",
    "--require-integer-tree-artifact",
    "--tree-fallback-mode ranked_valid_or_none",
    "--tree-selection-mode performance_compatible",
    "--action-validation-mode formal_v1",
    "--strict-candidate-deploy-cap",
    "--carry-over-aware-safety",
    "--lo-budget-overrun-guard-units 1",
)


def test_fixed_ranked_smoke_script_contains_hout_call() -> None:
    text = _script("run_viper_fixed_ranked_smoke.ps1")
    assert "scripts/evaluate_dqn_amc.py" in text
    for flag in REQUIRED_PROFILE_FLAGS:
        assert flag in text
    assert "semantic_validation_passed" in text
    assert "formal_v1_mask_step_mismatch_count" in text


def test_h2_h5_script_contains_two_evaluate_calls() -> None:
    text = _script("run_viper_fixed_ranked_h2_h5.ps1")
    assert text.count("scripts/evaluate_dqn_amc.py") >= 2


def test_h2_h5_script_contains_real_horizons() -> None:
    text = _script("run_viper_fixed_ranked_h2_h5.ps1")
    assert "--end-time 20000000" in text
    assert "--end-time 50000000" in text


def test_h2_h5_script_uses_isolated_output_dirs() -> None:
    text = _script("run_viper_fixed_ranked_h2_h5.ps1")
    assert 'Join-Path $OutputRoot "hout_h2"' in text
    assert 'Join-Path $OutputRoot "hout_h5"' in text


def test_h2_h5_script_requires_integer_ranked_profile() -> None:
    text = _script("run_viper_fixed_ranked_h2_h5.ps1")
    for flag in REQUIRED_PROFILE_FLAGS:
        assert flag in text
    assert "[string]$Seeds" in text
    assert "--seeds $Seeds" in text
    assert "semantic_validation_passed" in text
